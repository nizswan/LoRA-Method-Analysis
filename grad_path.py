import torch
import torchvision
from collections import OrderedDict
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda"

weights_dir = './RoBERTa/RoBERTaBASE/MRPC/ModelCheckpoints'
best_dir = './RoBERTa/RoBERTaBASE/MRPC/ModelCheckpointsBest/LoRA-R4-Best.pth'
file_prefix = 'LoRA-R4-Epoch'
out_dir = './RoBERTa/RoBERTaBASE/MRPC/ModelLandscapes/LoRA'
bestEpoch = 12

os.makedirs(out_dir, exist_ok=True)

# removed the ['model'] part since that needed to be removed before?
sample_state = torch.load(best_dir)

def normalize_directions_for_weights(direction:dict, weights:dict):
    assert (len(direction) == len(weights))
    dir_normed = OrderedDict()
    # for d, w in zip(direction, weights):
    for k, d in direction.items():
        if d.dim() <= 1:
            # d.fill_(0)
            dir_normed.update({k:torch.zeros_like(d)})
        # d.mul_(w.norm() / (d.norm() + 1e-10))
        dir_normed.update({k:torch.mul(d, (weights[k].norm() / (d.norm() + 1e-10)))})
    return dir_normed

def state2flat(state:dict):
    dim = []
    param = None
    for k, v in state.items():
        if param is not None:
            dim.append(v.to(device).reshape(-1).shape[0])
            param = torch.cat((param, v.to(device).reshape(-1)), dim=0)
        else:
            dim.append(v.to(device).reshape(-1).shape[0])
            param = v.to(device).reshape(-1)

    param = param.to(device)

    return param, dim

def flat2state(param:torch.Tensor, dim:list, state:dict=sample_state):
    assert len(state) == len(dim)
    assert len(param) == sum(dim)
    # wt = dict()
    wt = OrderedDict()
    counter = 0

    for i, (k, v) in enumerate(state.items()):
        assert dim[i] == v.numel()
        wt.update({k: param[counter:(counter + v.numel())].reshape(shape=v.shape)})
        counter += v.numel()
    
    assert counter == sum(dim)
    return wt

def bksvd(A:torch.Tensor, k:int=2, it:int=3, bsize:int=2, center:bool=False):
    if len(A.shape)>2: raise NotImplementedError('Only 2D Matrix allowed.')
    k = min(k, min(list(A.shape)))
    if k<1 or it<1 or bsize<k: raise ValueError('Incompatible Input Arguments! it>=1, k>=1, bsize>=k')

    A = A.to(device) # m x n

    u = torch.zeros(size=(1, A.shape[1])).to(device) # 1 x n
    if center:
        u = torch.mean(A, dim=0).to(device) # 1 x n
    
    l = torch.ones(size=(A.shape[0], 1)).to(device) # m x 1

    # n, ind = torch.min(torch.Tensor(list(A.shape))) # m is smaller dim
    ind = torch.argmin(torch.Tensor(list(A.shape))) # m is smaller dim
    tpose = False
    if ind == 0: # m < n True
        tpose = True
        l = torch.transpose(u, 0, 1) # n x 1
        u = torch.ones(size=(1, A.shape[0])).to(device) # 1 x m
        A = torch.transpose(A, 0, 1) # n x m
    
    krylov = torch.zeros(size=(A.shape[1], bsize*it)).to(device) # m x ki

    block = torch.randn(size=(A.shape[1], bsize)).to(device) # m x k

    block, R = torch.linalg.qr(block) # m x j, j x k

    T = torch.zeros(size=(A.shape[1], bsize)) # m x k

    for i in range(it):
        T = torch.mm(A, block) - torch.mm(l, torch.mm(u, block)) # n x j
        block = torch.mm(torch.transpose(A, 0, 1), T) - torch.mm(torch.transpose(u, 0, 1), (torch.mm(torch.transpose(l, 0, 1), T))) # m x j
        block, R = torch.linalg.qr(block) # m x j, j x k
        krylov[:, i*bsize:(i+1)*bsize] = block
    
    Q, R = torch.linalg.qr(krylov) # m x a, a x ki

    T = torch.mm(A, Q) - torch.mm(l, torch.mm(u, Q)) # n x a

    Ut, Std, Vth = torch.linalg.svd(T,full_matrices=False) # n x n, b, b x a
    Vt = torch.transpose(Vth, 0, 1) # a x b

    St = torch.diag(Std)
    S = St[0:k, 0:k] # k x k

    if not tpose:
        U = Ut[:, 0:k] # n x k
        V = torch.mm(Q, Vt[:, 0:k]) # m x k
    else:
        V = Ut[:, 0:k] # n x k
        U = torch.mm(Q, Vt[:, 0:k]) # m x k
    
    return U, V, S

param_list = []

best_param, dim = state2flat(sample_state)

#ckpt_list = ['neg1'] + list(range(9,100,10))
ckpt_list = list(range(1,bestEpoch+1,2))
print(ckpt_list)
# ckpt_list = ['neg1'] + list(range(19,100,20))

#for i in ckpt_list:
#    print("Completing adding checkpoint " + str(i))
#    file = file_prefix + '-' + str(i) + '.pth'
#    weights = torch.load((weights_dir + '/' + file), map_location='cuda')
#
#    # Removed ['model'] from weights['model']
#    param, d = state2flat(state=weights)
#    assert d == dim
#    param_list.append(param)
#
#params_all = torch.stack(param_list, dim=-1).to(device)
#del weights, param, param_list
#M = params_all - best_param.unsqueeze(-1)
#
#U, V, _ = bksvd(M, k=2,it=3, bsize=2)
#
#assert len(U.shape) == 2 and len(V.shape) == 2
#
#if U.shape[0] == sum(dim):
#    direction_set = U
#elif V.shape[0] == sum(dim):
#    direction_set = V
#else:
#    raise ValueError('Dimension mismatch')
#
#dir1 = flat2state(direction_set[:,0].squeeze(), dim=dim)
#dir2 = flat2state(direction_set[:,1].squeeze(), dim=dim)

#torch.save(dir1, os.path.join(out_dir, 'dirX.pth'))
#torch.save(dir2, os.path.join(out_dir, 'dirY.pth'))

#dir1_norm = normalize_directions_for_weights(dir1, sample_state)
#dir2_norm = normalize_directions_for_weights(dir2, sample_state)

#dir1_norm_f, _ = state2flat(dir1_norm)
#dir2_norm_f, _ = state2flat(dir2_norm)

#direction_set_normed = torch.stack((dir1_norm_f, dir2_norm_f), dim=1)

# grad_xy = torch.mm(torch.transpose(direction_set_normed, 0, 1), params_all)
grad_xy = torch.mm(torch.transpose(direction_set, 0, 1), M)
opt_xy = torch.tensor([0,0])
# opt_xy = torch.mm(torch.transpose(direction_set_normed, 0, 1), best_param.unsqueeze(-1))

torch.save({'weight_dist':params_all, 'XY':grad_xy, 'opt_XY':opt_xy}, os.path.join(out_dir, 'weight_dist.pth'))

####################
gparam_list = []
gckpt_list = list(range(1,bestEpoch+1,2))

for i in gckpt_list:
    print("Completing adding g-checkpoint " + str(i))
    file = file_prefix + '-' + str(i) + '.pth'
    weights = torch.load((weights_dir + '/' + file), map_location='cuda')

    # Removed ['model'] from the weights.
    param, d = state2flat(state=weights)
    assert d == dim
    gparam_list.append(param)

gparams_all = torch.stack(gparam_list, dim=-1).to(device)
gparams_meannorm = gparams_all - best_param.unsqueeze(-1)
grad_xy_f = torch.mm(torch.transpose(direction_set, 0, 1), gparams_meannorm)
torch.save({'FinalXY':grad_xy_f}, os.path.join(out_dir, 'final_grad_path.pth'))