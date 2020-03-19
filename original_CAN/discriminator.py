import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import helper

"""
Jose Ronaldo Pinheiro Carneiro Filho (jc4896)

implementation of original discriminator
"""

class Discriminator(nn.Module):
    
    def __init__(self, num_classes=1):
        super(Discriminator, self).__init__()
        
        v=256*256*3
        n=((v-4*4+2)/2+1)*32 #((input_volume−kernel_volume+2padding)/stride+1)*numberOfLayers
        
        self.conv1 = conv(v, n) #(32 4x4)
        self.conv2 = conv(n,2*n) #(64 4x4)
        self.conv3 = conv(2*n, 4*n) #(128 4x4)
        self.conv4 = conv(4*n, 8*n) #(256 4x4)
        self.conv5 = conv(8*n, 16*n) #(512 4x4)
        self.conv6 = conv(16*n, 16*n) #(512 4x4)
        
        self.multi1 = nn.Linear(16*n, 1024)
        self.multi2 = nn.Linear(1024, 512)
        self.multi3 = nn.Linear(512, num_classes)
        
        self real = nn.Linear(16*n, 2)

    #helper method to create convolutional layers
    def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    def forward(self, x):
        out = F.leaky_relu((self.conv1(x)), 0.1)
        out = F.leaky_relu((self.conv2(out)), 0.1)
        out = F.leaky_relu((self.conv3(out)), 0.1)
        out = F.leaky_relu((self.conv4(out)), 0.1)
        out = F.leaky_relu((self.conv5(out)), 0.1)
        out = F.leaky_relu((self.conv6(out)), 0.1)
        
        out.flatten()
        
        multi_output = self.multi1(out)
        multi_output = self.multi2(multi_output)
        multi_output = self.multi3(multi_output)
        multi_output = F.softmax(multi_output)
        
        real_output = self.real(out)
        real_output = self.sigmoid(real_output)
        
        return real_output, multi_output
