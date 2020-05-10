"""
Main training script
J.R. Carneiro JC4896
Yarne Hermann YPH2105
"""

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
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt

from data.stanford_dogs import StanfordDogs
from model import weights_init, Generator, Discriminator
from loss import G_loss, D_loss


####################################################
# These parameters should mainly be left unchanged #
####################################################
BATCH_SIZE = 32
ngpu = 1
num_workers = 4
lr_sgd = 0.0001
lr_adam = 0.0002
beta1_adam = 0.5
num_epochs = 500
CROP_SIZE = 128


# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
train_on_gpu = torch.cuda.is_available()


####################################
# These parameters can be changed  #
####################################

# The subset of dog classes to use. Set specific_classes to None to sue the entire Stanford Dogs dataset. 
# specific_classes = ["Maltese_dog"]
specific_classes = ["Maltese_dog", "Rhodesian_ridgeback", "bloodhound", "Norwegian_elkhound", "Staffordshire_bullterrier",
                    "standard_schnauzer", "cocker_spaniel", "Old_English_sheepdog", "Bouvier_des_Flandres", "Doberman"]

# If True, will load a model and continue training on it. 
# Provide timestamp and epoch. Timestamp will be reused.
# Epoch will continue incrementing from the provided value until num_epochs is reached
continue_training = False
continue_training_timestamp = '2020-05-03 18:13:26.043295'
continue_training_epoch = 300
# This value should not be changed
continue_training_path = 'models/{}_' + continue_training_timestamp + '_epoch_' + str(continue_training_epoch) + '_' + str(CROP_SIZE) + '.pt'

# How many updates to Genarator happen per update to Discriminator
G_iterations = 3

# Can be 'can', 'weighted_can' or 'dcgan'
mode = 'can'

# Can be set to 'adam' or 'sgd'
optimizer = 'adam'

# Extra updates on training are printed every print_every batches
print_every = 50

#################################################################



# Get dataloader
train_dataset = StanfordDogs('./images', resize=True, specific_classes=specific_classes, crop_size=CROP_SIZE)
print("Dataset size:", len(train_dataset), "Number of classes:", train_dataset.get_num_classes())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)


# Initialize model
if continue_training:
    G = torch.load(continue_training_path.format('G')).to(device)
    D = torch.load(continue_training_path.format('D')).to(device)
    dt = continue_training_timestamp
    start_epoch = continue_training_epoch
else:
    G = Generator(ngpu).to(device)
    D = Discriminator(ngpu, num_classes=train_dataset.NUM_CLASSES).to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    G.apply(weights_init)
    D.apply(weights_init)
    # Print the model
    print(G)
    print(D)
    dt = datetime.now()
    start_epoch = 0
    
# Set loss weights depending on mode  
if mode == 'can':
    D_loss_weights = None
    G_loss_weights = None
elif mode == 'weighted_can':
    D_loss_weights = [1.0, 0.1, 1.0]
    G_loss_weights = [1.0, 0.1]
elif mode == 'dcgan':
    D_loss_weights = [1.0, 0.0, 1.0]
    G_loss_weights = [1.0, 0.0]


# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_z = torch.randn(BATCH_SIZE, 100, 1, 1, device=device)

# Setup optimizers
if optimizer == 'adam':
    optimizerD = optim.Adam(D.parameters(), lr=lr_adam, betas=(beta1_adam, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr_adam, betas=(beta1_adam, 0.999))
elif optimizer == 'sgd':
    optimizerD = optim.SGD(D.parameters(), lr=lr_sgd)
    optimizerG = optim.SGD(G.parameters(), lr=lr_sgd)


# if train_on_gpu:
#     G.cuda()
#     D.cuda()
#     print('GPU available for training. Models moved to GPU')
# else:
#     print('Training on CPU.')


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []


print("Starting Training Loop...")

for epoch in range(start_epoch, num_epochs):
    for batch_i, (real_images, real_labels) in enumerate(train_dataloader):
#         print(batch_i, '/', len(train_dataloader))
        info = (batch_i % print_every == 0)
        b_size = real_images.size(0)
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        D.zero_grad()
        
#         if train_on_gpu:
#             real_images = real_images.cuda()
        real_images = real_images.to(device)
        D_real, D_multi = D(real_images) 
                
        # Generate fake image batch with G
        z = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = G(z)
        
#         if train_on_gpu:
#             real_images = real_images.cuda()
        D_fake, _ = D(fake_images)    
            
        # Calculate loss and update D
        d_loss = D_loss(D_real, D_multi, real_labels, D_fake, device, weights=D_loss_weights, info=info)
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        for it in range(G_iterations):
            G.zero_grad()
            
            # Generate fake image batch with G
            z = torch.randn(b_size, 100, 1, 1, device=device)
            fake_images = G(z)
            D_fake, D_fake_entropy = D(fake_images) 

            # Calculate loss and update G
            g_loss = G_loss(D_fake, D_fake_entropy, device, weights=G_loss_weights, info=info)        
            g_loss.backward()
            optimizerG.step()
        
        
        ######################################################################
    
#         print('D Loss:', d_loss.data.cpu().numpy(), 'G Loss:', g_loss.data.cpu().numpy())
        # Output training stats
        if info:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}], batch {} | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, batch_i, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##  
    
    # append discriminator loss and generator loss
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())
    # Every 20 epochs: backup to disk so we can load and visualize
    if (epoch+1) % 20 == 0:
        if not continue_training:
            with open('losses/G_losses' + str(dt) + '_' + str(CROP_SIZE) + '.pkl', 'wb') as f:
                pkl.dump(G_losses, f)
            with open('losses/D_losses' + str(dt) + '_' + str(CROP_SIZE) + '.pkl', 'wb') as f:
                pkl.dump(D_losses, f)
        else: 
            with open('losses/G_losses' + str(dt) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pkl', 'wb') as f:
                pkl.dump(G_losses, f)
            with open('losses/D_losses' + str(dt) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pkl', 'wb') as f:
                pkl.dump(D_losses, f)
    
    
    # generate and save sample, fake images for the fixed_z
    G.eval() # for generating samples
#     if train_on_gpu:
#         fixed_z = fixed_z.cuda()
#     fixed_z = fixed_z.to(device)
    img_z = G(fixed_z).detach().cpu()
    img_list.append(img_z)
    # Save training generator samples
    if continue_training:
        sample_save_path = 'img_z/img_z_' + str(dt) + '_epoch_' + str(epoch+1) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pkl'
    else:
        sample_save_path = 'img_z/img_z_' + str(dt) + '_epoch_' + str(epoch+1) + '_' + str(CROP_SIZE) + '.pkl'
    print("SAVING:", sample_save_path)
    with open(sample_save_path, 'wb') as f:
        pkl.dump(img_z, f)
    
    #img_list.append(make_grid(img_z, padding=2, normalize=True))
    G.train() # back to training mode  
    
    # Save models
    if continue_training:
        model_save_path = 'models/{}_' + str(dt) + '_epoch_' + str(epoch+1) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pt'
    else:
        model_save_path = 'models/{}_' + str(dt) + '_epoch_' + str(epoch+1) + '_' + str(CROP_SIZE) + '.pt'
    torch.save(G, model_save_path.format('G'))
    torch.save(D, model_save_path.format('D'))


# Make sure to save all losses after training has finished
if not continue_training:
    with open('losses/G_losses' + str(dt) + '_' + str(CROP_SIZE) + '.pkl', 'wb') as f:
        pkl.dump(G_losses, f)
    with open('losses/D_losses' + str(dt) + '_' + str(CROP_SIZE) + '.pkl', 'wb') as f:
        pkl.dump(D_losses, f)
else: 
    with open('losses/G_losses' + str(dt) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pkl', 'wb') as f:
        pkl.dump(G_losses, f)
    with open('losses/D_losses' + str(dt) + '_' + str(CROP_SIZE) + '_continue_epoch_' + str(continue_training_epoch) + '.pkl', 'wb') as f:
        pkl.dump(D_losses, f)
    
    

    
    