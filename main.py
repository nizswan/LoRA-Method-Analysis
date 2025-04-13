# For MRPC dataset RoBERTa
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_FFT as RoBERTaBASE_MRPC_FFT
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_LoRA as RoBERTaBASE_MRPC_LoRA
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_CoLA as RoBERTaBASE_MRPC_CoLA
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_Cheap as RoBERTaBASE_MRPC_Cheap
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_FixA as RoBERTaBASE_MRPC_FixA
import RoBERTa.RoBERTaBASE.MRPC.RoBERTaBASE_MRPC_RAC as RoBERTaBASE_MRPC_RAC

# For CoLA dataset RoBERTa
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_FFT as RoBERTaBASE_CoLA_FFT
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_LoRA as RoBERTaBASE_CoLA_LoRA
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_CoLA as RoBERTaBASE_CoLA_CoLA
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_Cheap as RoBERTaBASE_CoLA_Cheap
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_FixA as RoBERTaBASE_CoLA_FixA
import RoBERTa.RoBERTaBASE.CoLA.RoBERTaBASE_CoLA_RAC as RoBERTaBASE_CoLA_RAC


# def Run(maxLength=288, batchSize=8, learningRate, epochs, chainReset, rank, alpha, projectName):
# For MRPC dataset RoBERTa
#RoBERTaBASE_MRPC_FFT.Run(maxLength=288, batchSize=16, learningRate=2e-5, epochs=100, chainReset=0, rank=2, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_MRPC_LoRA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=4, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_MRPC_CoLA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=10, rank=4, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_MRPC_Cheap.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=8, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_MRPC_FixA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=8, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_MRPC_RAC.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=10, rank=8, alpha=0, projectName="LoRA_Data_Collection")


# For CoLA dataset RoBERTa
RoBERTaBASE_CoLA_FFT.Run(maxLength=288, batchSize=16, learningRate=2e-5, epochs=100, chainReset=0, rank=2, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_CoLA_LoRA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=4, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_CoLA_CoLA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=10, rank=4, alpha=0, projectName="LoRA_Data_Collection")

#RoBERTaBASE_CoLA_Cheap.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=8, alpha=0, projectName="LoRA_Data_Collection")

RoBERTaBASE_CoLA_FixA.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=0, rank=8, alpha=0, projectName="LoRA_Data_Collection")

RoBERTaBASE_CoLA_RAC.Run(maxLength=288, batchSize=8, learningRate=4e-4, epochs=100, chainReset=10, rank=8, alpha=0, projectName="LoRA_Data_Collection")


