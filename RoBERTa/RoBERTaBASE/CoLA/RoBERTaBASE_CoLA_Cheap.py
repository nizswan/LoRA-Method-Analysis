from datasets import *
from transformers import RobertaModel, RobertaTokenizer, DataCollatorWithPadding

import torch
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader, Dataset

# adding wanb
import wandb
import time
import math

import numpy as np

# defaults are: maxLenghth=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=4, alpha=0, projectName="LoRA_Data_Collection"
def Run(maxLength, batchSize, learningRate, epochs, chainReset, rank, alpha, projectName):
    print("Running RoBERTaBASE adapted to CoLA dataset via Cheap-LoRA, rank=" + str(rank) +" alpha=" +str(alpha) + " epochs=" +str(epochs) + " lr=" +str(learningRate) + " batchSize=" +str(batchSize))
    ds = load_dataset("glue", "cola")

    TRAIN_SUBSET_SIZE = 8551
    TEST_SUBSET_SIZE = 1063
    VALID_SUBSET_SIZE = 1043

    train_dataset = ds['train'].shuffle(seed=46).select(range(TRAIN_SUBSET_SIZE))
    test_dataset = ds['test'].shuffle(seed=46).select(range(TEST_SUBSET_SIZE))
    valid_dataset = ds['validation'].shuffle(seed=46).select(range(VALID_SUBSET_SIZE))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
    
    def tokenize_text(batch):
        return tokenizer(batch["sentence"],
                        padding=True,
                        truncation=True,
                        return_token_type_ids=True,
                        max_length=maxLength)
    
    tokenized_train_dataset = train_dataset.map(tokenize_text, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_text, batched=True)
    tokenized_valid_dataset = valid_dataset.map(tokenize_text, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    del ds
    
    columns=["label", "input_ids", "attention_mask", "token_type_ids"]
    
    tokenized_train_dataset.set_format("torch", columns=columns)
    tokenized_test_dataset.set_format("torch", columns=columns)
    tokenized_valid_dataset.set_format("torch", columns=columns)
    
    class MyDataset(Dataset):
        def __init__(self, dataset, partition_key):
            self.dataset = dataset
    
        def __getitem__(self, index):
            return self.dataset[index]
    
        def __len__(self):
            return self.dataset.num_rows
        
    train_data = MyDataset(tokenized_train_dataset, partition_key="train")
    test_data = MyDataset(tokenized_test_dataset, partition_key="test")
    valid_data = MyDataset(tokenized_valid_dataset, partition_key="valid")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(dataset=test_data, batch_size=batchSize, collate_fn=data_collator)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batchSize, collate_fn=data_collator)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    class RobertaWithClassification(torch.nn.Module):
        def __init__(self):
            super(RobertaWithClassification, self).__init__()
            self.roberta = RobertaModel.from_pretrained("roberta-base")
            self.linear = torch.nn.Linear(768, 768)
            self.activation = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(768, 2)
        
        def forward(self, input_ids, attention_mask, token_type_ids):
            output_with_pooling = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden_state = output_with_pooling[0]
            pooler = hidden_state[:,0]
            pooler = self.linear(pooler)
            pooler = self.activation(pooler)
            pooler = self.dropout(pooler)
            output = self.classifier(pooler)
            return output
        
    model = RobertaWithClassification()
    model.to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    base_param_count = count_parameters(model)
    print(base_param_count)
        
    lr = learningRate
    EPOCHS = epochs
    CHAIN_RESET = chainReset
    
    lora_r = rank
    lora_alpha = 1
    if(alpha != 0):
        lora_alpha = alpha
    else:
        loha_alpha = lora_r * 2
    
    project_name = projectName
    run_name = "RoBERTaBase-CoLA-Cheap-R" + str(lora_r)
    
    run = wandb.init(name=run_name, project=project_name)
    
    def get_accuracy(y_pred, targets):
        predictions = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
        accuracy = (predictions == targets).sum() / len(targets)
        return accuracy
    
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    
    
    def train(model, train_loader, epochs, optimizer, chain_reset):
        decay = lr / epochs
        total_time = 0
        for g in optimizer.param_groups:
            CurRate = g['lr']
        print("Epochs: ", epochs)
        chain_count = 1
        best_epoch = 0
        best_corr = 0
        best_val = 9999
        for epoch in range(epochs):
            if(epoch != 0):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] - decay
                    print("lr ", g['lr'])
                    CurRate = g['lr']
            interval = len(train_loader) // 5
            
            strName = './RoBERTa/RoBERTaBASE/CoLA/ModelCheckpoints/Cheap-R' + str(lora_r) + '-Epoch-' + str(epoch) + '.pth'
            strNameTwo = ""
            if (epoch == 0):
                strNameTwo = './RoBERTa/RoBERTaBASE/CoLA/ModelCheckpointsBest/Cheap-R' + str(lora_r) + '-Epoch-' + str(epoch) + '.pth'
            torch.save(model.state_dict(), strName)
            if (epoch == 0):
                torch.save(model.state_dict(), strNameTwo)
            
            total_train_loss = 0
            total_train_acc = 0
            
            start = time.time()
            
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
                
                loss = loss_function(outputs, labels)
                acc = get_accuracy(outputs, labels)
                
                total_train_loss += loss.item()
                total_train_acc += acc.item()
                
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % interval == 0:
                    print("Batch: %s/%s | Training loss: %.4f | accuracy: %.4f" % (batch_idx+1, len(train_loader), loss, acc))
                    
            train_loss = total_train_loss / len(train_loader)
            train_acc = total_train_acc / len(train_loader)
            
            end = time.time()
            hours, remainder = divmod(end - start, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            val_data = evaluate(model, valid_loader, 2)
            #test_data = evaluate(model, test_loader, 1)
            
            if((val_data[2] > best_corr) and (epoch+1 in range(epochs))):
                best_epoch = epoch+1
                best_corr = val_data[2]
                best_val = val_data[1]
            
            run.log(
               {
                    "train/accuracy": float(f"{train_acc:.4f}"),
                    "train/loss": float(f"{train_loss:.4f}"),
                    "epoch": epoch+1,
                    "time": end - start,
                    "val/accuracy": float(f"{val_data[0]:.4f}"),
                    "val/loss": float(f"{val_data[1]:.4f}"),
                    "val/corr": float(f"{val_data[2]:.4f}"),
                    #"test/accuracy": float(f"{test_data[0]:.4f}"),
                    #"test/loss": float(f"{test_data[1]:.4f}"),
               }
            )
            
            print(f"Epoch: {epoch+1} train loss: {train_loss:.4f} train acc: {train_acc:.4f}")
            print("Epoch time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            print("")
            
            if(val_data[1] - best_val > 0.2):
                print("Reached a validation loss that is at least 0.2 beyond the best validation loss, ending training early")
                break
            
            total_time += (end - start)
            
        

    
        average_time_per_epoch = total_time /  epochs
        hours, remainder = divmod(average_time_per_epoch, 3600)
        minutes, seconds = divmod(remainder, 60)
    
        print("Average time per epoch: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        bestLocation = './RoBERTa/RoBERTaBASE/CoLA/ModelCheckpoints/Cheap-R' + str(lora_r) + '-Epoch-' + str(best_epoch) + '.pth'
        checkpoint = torch.load(bestLocation, map_location='cpu')
        model.load_state_dict(checkpoint)
        bestName = './RoBERTa/RoBERTaBASE/CoLA/ModelCheckpointsBest/Cheap-R' + str(lora_r) + '-Best.pth'
        torch.save(model.state_dict(), bestName)
        
        return best_epoch
    
    
    def evaluate(model, test_loader, version):
        interval = len(test_loader) // 5
        
        total_test_loss = 0
        total_test_acc = 0
        
        # Used for calculating Matthews Correlation Coefficient
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        model.eval()
        with torch.no_grad(): 
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                loss = loss_function(outputs, labels)
                acc = get_accuracy(outputs, labels)
                
                total_test_loss += loss.item()
                total_test_acc += acc.item()
                
                
                corrOutputs = np.argmax(outputs.cpu(), axis=1)
                labelsOutputs = labels.cpu()
                for k in range(len(corrOutputs)):
                    if(corrOutputs[k] == 1 and labelsOutputs[k] == 1):
                        TP+=1
                    if(corrOutputs[k] == 1 and labelsOutputs[k] == 0):
                        FP+=1  
                    if(corrOutputs[k] == 0 and labelsOutputs[k] == 0):
                        TN+=1
                    if(corrOutputs[k] == 0 and labelsOutputs[k] == 1):
                        FN+=1
                
                if (batch_idx + 1) % interval == 0:
                    if(version == 1):
                        print("Batch: %s/%s | Test loss: %.4f | accuracy: %.4f" % (batch_idx+1, len(test_loader), loss, acc))
                    elif(version == 2):
                        print("Batch: %s/%s | Valid loss: %.4f | accuracy: %.4f" % (batch_idx+1, len(test_loader), loss, acc))
        
        print("TP ", TP, " TN ", TN, " FP ", FP, " FN ", FN)
        corrCoeff = 5
        if((TP+FP != 0) and (TN+FN != 0) and (TN+FP != 0) and (TN+FN != 0)):
            corrCoeff = (TP * TN - FP * FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        else:
            corrCoeff = -10
        
        test_loss = total_test_loss / len(test_loader)
        test_acc = total_test_acc / len(test_loader)
        
        data = [test_acc, test_loss, corrCoeff]
    
        if(version == 1):
            print(f"Test loss: {test_loss:.4f} acc: {test_acc:.4f} corr: {corrCoeff:.4f}")
            print("")
        elif(version == 2):
            print(f"Valid loss: {test_loss:.4f} acc: {test_acc:.4f} corr: {corrCoeff:.4f}")
            print("")
    
        return data
    
    
    # Modified for Cheap LoRA, fix A and keep original as eye.
    class LoRALayer(torch.nn.Module):
        def __init__(self, original_layer, rank, alpha):
            super().__init__()
            self.original_layer = original_layer
            self.rank = rank
            self.alpha = alpha
            self.lora_A = torch.nn.Linear(original_layer.in_features, rank, bias=False)
            for param in self.lora_A.parameters():
                param.requires_grad = False
            self.lora_B = torch.nn.Linear(rank, original_layer.out_features, bias=False)
            torch.nn.init.eye_(self.lora_A.weight)
            torch.nn.init.zeros_(self.lora_B.weight)
        
        def forward(self, x):
            return self.original_layer(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
        
        
    lora_model = RobertaWithClassification()
    
    for param in lora_model.parameters():
        param.requires_grad = False
        
    for layer in lora_model.roberta.encoder.layer:
        layer.attention.self.query = LoRALayer(layer.attention.self.query, lora_r, lora_alpha)
        layer.attention.self.value = LoRALayer(layer.attention.self.value, lora_r, lora_alpha)
        
    
    lora_param_count = count_parameters(lora_model)
    print("Model with LoRA param count:", lora_param_count)
    print("Base model param count:", base_param_count)
    print(str(base_param_count // lora_param_count) + " times smaller than base model")
    
    run.log(
        {
            "param/count": lora_param_count,
            "param/times_smaller": base_param_count // lora_param_count
        }
    )
    
    lora_model.to(device)
    
    optimizer_lora = torch.optim.Adam(params=lora_model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()
    
    bestEpoch = train(lora_model, train_loader, EPOCHS, optimizer_lora, CHAIN_RESET)
    
    #result = evaluate(lora_model, test_loader,1)
    result_val = evaluate(lora_model, valid_loader,2)
    
    run.log(
        {
            #"bestModel/test/accuracy": float(f"{result[0]:.4f}"),
            #"bestModel/test/loss": float(f"{result[1]:.4f}"),
            "bestModel/val/accuracy": float(f"{result_val[0]:.4f}"),
            "bestModel/val/loss": float(f"{result_val[1]:.4f}"),
            "bestModel/val/corr": float(f"{result_val[2]:.4f}"),
            "bestModel/epoch": float(f"{bestEpoch:.4f}"),
        }
    )
    
    run.finish()

    
