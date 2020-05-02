import torch.nn as nn
import torch.nn.functional as F

"""
Custom weight initialization called on Generator and Discriminator
(from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


        
        
"""
Implementation of original generator inspired by 
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html and CAN paper
J.R. Carneiro JC4896
Yarne Hermann YPH2105
"""
class Generator(nn.Module):
    
    def __init__(self, ngpu):
        super(Generator, self).__init__()
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
            nn.ConvTranspose2d( 128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 128 x 128
        )
        
    def forward(self, x):
        return self.main(x)



"""
Implementation of original discriminator inspired by 
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html and CAN paper
J.R. Carneiro JC4896
Yarne Hermann YPH2105
"""
class Discriminator(nn.Module):
    
    def __init__(self, ngpu, num_classes=120):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        # input is 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1, bias=False) 
        # state size. 32 x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False) 
        self.bn2 = nn.BatchNorm2d(64)
        # state size. 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # state size. 128 x 16 x 16
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        # state size. 256 x 8 x 8
        self.conv5 = nn.Conv2d(256, 256, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Real/Fake output
        self.real = nn.Linear(256 * 4 * 4, 1)
        
        # Multi-label output
        self.multi1 = nn.Linear(256 * 4 * 4, 1024)
        self.multi2 = nn.Linear(1024, 512)
        self.multi3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        shared_out = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn2(self.conv2(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn3(self.conv3(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn4(self.conv4(shared_out)), 0.2, inplace=True)
        shared_out = F.leaky_relu(self.bn5(self.conv5(shared_out)), 0.2, inplace=True)
        shared_out = shared_out.view(-1, 256 * 4 * 4)

        real_output = F.sigmoid(self.real(shared_out))

        multi_output = self.multi1(shared_out)
        multi_output = self.multi2(multi_output)
        multi_output = F.softmax(self.multi3(multi_output))

        return real_output, multi_output


