import os
import random
import time
from datetime import datetime
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data.stanford_dogs import StanfordDogs

train_on_gpu = torch.cuda.is_available()


BATCH_SIZE = 128
ngpu = 1
lr = 0.0001
num_epochs = 100
CROP_SIZE = 256


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")






"""
Yarne Hermann YPH2105
"""

train_dataset = StanfordDogs('./images', crop_size=crop_size, resize=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


# custom weights initialization called on netG and netD
# (from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


        
        
"""
implementation of original generator
"""
"""
J.R. Carneiro JC4896
Yarne Hermann YPH2105
"""

class Generator(nn.Module):
    
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        
        '''
        The following is inspired by 
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        which seemed a bit clearer and from the CAN paper
        '''
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            # state size. 2048 x 4 x 4
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 8 x 8
            nn.ConvTranspose2d( 1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            nn.ConvTranspose2d( 512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 32 x 32
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 64 x 64
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 128 x 128
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 256 x 256
        )
        
    def forward(self, x):
        return self.main(x)





"""
implementation of original discriminator
"""
"""
J.R. Carneiro JC4896
Yarne Hermann YPH2105
"""

class Discriminator(nn.Module):
    
    def __init__(self, ngpu, num_classes=120):
        super(Discriminator, self).__init__()
        
        '''
        The following is inspired by 
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        which seemed a bit clearer and from the CAN paper
        '''
        
        self.ngpu = ngpu
        self.num_classes = num_classes
        # input is 3 x 256 x 256
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1, bias=False) 
        # state size. 32 x 128 x 128
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False) 
        self.bn2 = nn.BatchNorm2d(64)
        # state size. 64 x 64 x 64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # state size. 128 x 32 x 32
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        # state size. 256 x 16 x 16
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        # state size. 512 x 8 x 8
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.real = nn.Linear(512 * 4 * 4, 1)
        
        self.multi1 = nn.Linear(512 * 4 * 4, 1024)
        self.multi2 = nn.Linear(1024, 512)
        self.multi3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        shared_out = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn2(self.conv2(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn3(self.conv3(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn4(self.conv4(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn5(self.conv5(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn6(self.conv6(shared_out)), 0.2, inplace=True)
        

        shared_out = shared_out.view(-1, 512 * 4 * 4)

        real_output = F.sigmoid(self.real(shared_out))

        multi_output = self.multi1(shared_out)
        multi_output = self.multi2(multi_output)
        multi_output = F.softmax(self.multi3(multi_output))

        return real_output, multi_output







def D_loss(D_out_real, D_out_multi, multi_labels, D_out_false, weights=None, info=False):
    batch_size = D_out_multi.size(0)
    eps=1e-7
    log_r = torch.mean(torch.log(D_out_real + eps))
    
    row_indices = torch.from_numpy(np.arange(batch_size))
    if train_on_gpu:
        multi_labels = multi_labels.cuda()
        row_indices = row_indices.cuda()
    
    
    multi_outputs = D_out_multi[row_indices, multi_labels]
    log_m = torch.mean(torch.log(multi_outputs+eps))
    log_f = torch.mean(torch.log(1 - D_out_false + eps))
    
    if info:
        print("DRR", log_r.data.cpu().numpy(), "DRM", log_m.data.cpu().numpy(), "DFF", log_f.data.cpu().numpy())
    if weights is None: 
        return - (log_r + log_m + log_f)
    else:
        return - (weights[0] * log_r + weights[1] * log_m + weights[2] * log_f )
#         s = sum(weights)
#         return - (weights[0]/s * log_r + weights[1]/s * log_m + weights[2]/s * log_f )




"""
Yarne Hermann YPH2105
"""
# Have to make sure to be correct about maximizing or minimizing loss.
# I took the negative of what is mentioned on page 9 in the paper in order to create a loss
# to be minimized. If I'm correct real_loss can be used as it is right now
def entropy_loss(D_out):
    eps=1e-7
    batch_size = D_out.size(0)
    K = D_out.size(1)
    loss = torch.zeros(batch_size)
    if train_on_gpu:
        D_out = D_out.cuda()
        loss = loss.cuda()
            
    for c in range(K):
        
        
        # c_loss = 1/K * torch.log(probabilities[:, c]) + (1 - 1/K) * torch.log(torch.ones(batch_size)-probabilities[:, c])         
        c_loss = 1/K * torch.log(D_out[:, c] + eps) + (1 - 1/K) * torch.log(1-D_out[:, c]+eps)         
        
        loss += c_loss
    #print(loss)
    return loss.mean()




def G_loss(D_out_false, D_out_multi, weights=None, info=False):
    batch_size = D_out_multi.size(0)
    eps=1e-7
    log_f = torch.mean(torch.log(D_out_false + eps))
    l_entropy = entropy_loss(D_out_multi)
    
    if info:
        print("GFR", log_f.data.cpu().numpy(), "GFE", l_entropy.data.cpu().numpy())
    

    if weights is None: 
        return  - (log_f + l_entropy)
    else:
        return - (weights[0] * log_f + weights[1] * l_entropy)
#         s = sum(weights)
#         return - (weights[0]/s * log_f + weights[1]/s * l_entropy)





G = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    G = nn.DataParallel(G, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
G.apply(weights_init)

# Print the model
print(G)

# Create the Discriminator
D = Discriminator(ngpu, num_classes=train_dataset.NUM_CLASSES).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    D = nn.DataParallel(D, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
D.apply(weights_init)

# Print the model
print(D)


# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_z = torch.randn(BATCH_SIZE, 100, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.SGD(D.parameters(), lr=lr)
optimizerG = optim.SGD(G.parameters(), lr=lr)


if train_on_gpu:
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')




# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
print_every = 50
dt = datetime.now()
D_loss_weights = [1.0, 0.1, 1.0]
G_loss_weights = [1.0, 0.1]

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for batch_i, (real_images, real_labels) in enumerate(train_dataloader, 0):
        print(batch_i, '/', len(train_dataloader))
        info = (batch_i % print_every == 0)
#         info=True
        b_size = real_images.size(0)
        optimizerG.zero_grad()
        
        # 3.
        #z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100)) 
        #z = torch.from_numpy(z).float()
        z = torch.randn(b_size, 100, 1, 1, device=device)
        #if train_on_gpu:
        #    z = z.cuda()
        
        # 4) Generate fake image batch with G
        fake_images = G(z)
        
        if train_on_gpu:
            real_images = real_images.cuda()
        
        # 5) Forward pass real batch through D
        D_real, D_multi = D(real_images) #.view(-1)
#         d_real_real_loss = real_loss(D_real, label=1) 
#         # 6.
#         d_real_multi_loss = multi_loss(D_multi, real_labels)
        # 7.
        D_fake, D_fake_entropy = D(fake_images)
#         d_fake_real_loss = real_loss(D_fake, label=0)
#         g_fake_real_loss = real_loss(D_fake, label=1)
#         # 8.
#         g_fake_entropy_loss = entropy_loss(D_fake_entropy) ##
        
        # 9.
        #d_loss= torch.log(d_real_real_loss)+torch.log(d_real_multi_loss)+torch.log(d_fake_real_loss) 
        #d_loss = d_real_real_loss + d_real_multi_loss + d_fake_real_loss
        d_loss = D_loss(D_real, D_multi, real_labels, D_fake, weights=D_loss_weights, info=True)
        
        #torch.log(1-g_fake_real_loss), the 1- is not necessary because computed against label=0 now
#         print('DRR Loss:', d_real_real_loss.data.cpu().numpy(),
#               'DRM Loss:', d_real_multi_loss.data.cpu().numpy(), 
#               'DFR Loss:',d_fake_real_loss.data.cpu().numpy(), 
#               'D Loss:',d_loss.data.cpu().numpy())
        
        # 10.
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        
        # 11.
        #g_loss=torch.log(g_fake_real_loss)-g_fake_entropy_loss
        #g_loss=g_fake_real_loss - g_fake_entropy_loss
        g_loss = G_loss(D_fake, D_fake_entropy, weights=G_loss_weights, info=True)
        
#         print('GFR Loss:',g_fake_real_loss.data.cpu().numpy(), 
#               'GFE Loss:',g_fake_entropy_loss.data.cpu().numpy(), 
#               'G Loss:',g_loss.data.cpu().numpy())
        
        # 12.
        g_loss.backward()
        optimizerG.step()
        print('D Loss:', d_loss.data.cpu().numpy(), 'G Loss:', g_loss.data.cpu().numpy())

        
        # Output training stats
        if info:
            # append discriminator loss and generator loss
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}], batch {} | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, batch_i, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##    
    # generate and save sample, fake images
    G.eval() # for generating samples
    if train_on_gpu:
        fixed_z = fixed_z.cuda()
    img_z = G(fixed_z).detach().cpu()
    img_list.append(img_z)
    # Save training generator samples
    with open('img_z/img_z_' + str(dt) + '_epoch_' + str(epoch+1) + '.pkl', 'wb') as f:
        pkl.dump(img_z, f)
    
    #img_list.append(make_grid(img_z, padding=2, normalize=True))
    G.train() # back to training mode  
    
    torch.save(G, 'models/G_' + str(dt) + '_epoch_' + str(epoch+1) + '.pt')
    torch.save(D, 'models/D_' + str(dt) + '_epoch_' + str(epoch+1) + '.pt')
    
with open('losses/G_losses' + str(dt) + '.pkl', 'wb') as f:
        pkl.dump(G_losses, f)
with open('losses/D_losses' + str(dt) + '.pkl', 'wb') as f:
        pkl.dump(D_losses, f)
    
# # Save training generator samples
# with open('train_samples.pkl', 'wb') as f:
#     pkl.dump(img_list, f)
    



### END -   FROM Udacity DCGAN implementation ###






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





