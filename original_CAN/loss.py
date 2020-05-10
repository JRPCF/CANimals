import torch
import numpy as np

"""
Entropy loss inspired by CAN paper. Part of G_loss
Yarne Hermann YPH2105

params:
D_out: Multi-label output of Discriminator
device: Device to move tensors to
"""
def entropy_loss(D_out, device):
    eps=1e-7
    batch_size = D_out.size(0)
    K = D_out.size(1)
    loss = torch.zeros(batch_size)

    D_out = D_out.to(device)
    loss = loss.to(device)
            
    for c in range(K):        
        c_loss = 1/K * torch.log(D_out[:, c] + eps) + (1 - 1/K) * torch.log(1-D_out[:, c]+eps)         
        loss += c_loss

    return loss.mean()



"""
Discriminator loss function inspired by CAN paper
J.R. Carneiro JC4896
Yarne Hermann YPH2105

params:
D_out_real: Real/fake output of Discriminator on batch of real images
D_out_multi: Multi-label output of Discriminator on batch of real images
multi_labels: Correct class labels of images in batch of real images
D_out_false: Real/fake output of Discriminator on batch of generated images
device: Device to move tensors to

weights: (list of 3 floats) Custom weights to be applied to the 3 different parts of the total loss
info: (boolean) Flag to print the 3 individual losses
"""
def D_loss(D_out_real, D_out_multi, multi_labels, D_out_false, device, weights=None, info=False):
    batch_size = D_out_multi.size(0)
    eps=1e-7
    log_r = torch.mean(torch.log(D_out_real + eps))
    
    row_indices = torch.from_numpy(np.arange(batch_size))
    
    multi_labels = multi_labels.to(device)
    row_indices = row_indices.to(device)
    
    
    multi_outputs = D_out_multi[row_indices, multi_labels]
    log_m = torch.mean(torch.log(multi_outputs+eps))
    log_f = torch.mean(torch.log(1 - D_out_false + eps))
    
    if info:
        print("DRR", log_r.data.cpu().numpy(), "DRM", log_m.data.cpu().numpy(), "DFF", log_f.data.cpu().numpy())
    if weights is None: 
        return - (log_r + log_m + log_f)
    else:
        return - (weights[0] * log_r + weights[1] * log_m + weights[2] * log_f )


"""
Generator loss function inspired by CAN paper
J.R. Carneiro JC4896
Yarne Hermann YPH2105

params:
D_out_false: Real/fake output of Discriminator on batch of generated images
D_out_multi: Multi-label output of Discriminator on batch of real images
device: Device to move tensors to

weights: (list of 2 floats) Custom weights to be applied to the 2 different parts of the total loss
info: (boolean) Flag to print the 2 individual losses
"""
def G_loss(D_out_false, D_out_multi, device, weights=None, info=False):
    eps=1e-7
    log_f = torch.mean(torch.log(D_out_false + eps))
    l_entropy = entropy_loss(D_out_multi, device)
    
    if info:
        print("GFR", log_f.data.cpu().numpy(), "GFE", l_entropy.data.cpu().numpy())
    
    if weights is None: 
        return  - (log_f + l_entropy)
    else:
        return - (weights[0] * log_f + weights[1] * l_entropy)



