import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import helper

"""
Jose Ronaldo Pinheiro Carneiro Filho (jc4896)

implementation of original generator
"""

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        conv_dim = 256*256*3
        
        self.fc = nn.Linear(100, conv_dim*2*2*2*2*2*2*2)

        self.t_conv1 = deconv(conv_dim*2*2*2*2*2*2*2, conv_dim*2*2*2*2*2*2)
        self.t_conv2 = deconv(conv_dim*2*2*2*2*2*2, conv_dim*2*2*2*2*2)
        self.t_conv3 = deconv(conv_dim*2*2*2*2*2, conv_dim*2*2*2*2)
        self.t_conv4 = deconv(conv_dim*2*2*2*2, conv_dim*2*2*2)
        self.t_conv5 = deconv(conv_dim*2*2*2, conv_dim*2*2)
        self.t_conv6 = deconv(conv_dim*2*2, conv_dim*2)
        self.t_conv7 = deconv(conv_dim*2, conv_dim)
        
    #helper method to create deconvolutional layers
    def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.conv_transpose2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 2048) 
        
        out = F.leaky_relu(t_conv1(out),0.1)
        out = F.leaky_relu(t_conv2(out),0.1)
        out = F.leaky_relu(t_conv3(out),0.1)
        out = F.leaky_relu(t_conv4(out),0.1)
        out = F.leaky_relu(t_conv5(out),0.1)
        out = F.leaky_relu(t_conv6(out),0.1)
        out = F.leaky_relu(t_conv7(out),0.1)
        
        out = F.tanh(out)
        
        return out