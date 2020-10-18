import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock


class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block = None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if basic_block is not None:
            self.res_block = basic_block
        else:
            self.res_block = None

    def forward(self, x):
        x = self.conv1(x)
        if self.res_block is not None:
            r1 = self.res_block(x)
            x = x + r1

        return x

class Modified_Resnet(nn.Module):
    def __init__(self, residual_block, resnet_base):
        super(Modified_Resnet, self).__init__()
        #Prep Layer
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,  stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = residual_block(64, 128, basic_block=resnet_base(128, 128))
        self.layer2 = residual_block(128, 256)
        self.layer3 = residual_block(256, 512, basic_block=resnet_base(512, 512))
        self.pool1 = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10, bias=False)
        
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 

def call_model():
    return Modified_Resnet(Residual_Block, BasicBlock)       
