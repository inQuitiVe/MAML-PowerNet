import torch 
import torch.nn as nn
import numpy as np
from math import ceil

    
class PowerNet(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size = 3):
        

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size)

        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(8*ceil(in_channels/4)*ceil(in_channels/4), 64)
        self.fc2 = nn.Linear(64, out_channels)
        
    def forward(self,x):

        # Input CNN Layer  K*K*in_ch ==> (K/2)*(K/2)*16
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        X = self.pool(x)

        # CNN Layer 2  (K/2)*(K/2)*16 ==> (K/4)*(K/4)*16
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        X = self.pool(x)

        # CNN Layer 3  (K/4)*(K/4)*16 ==> (K/4)*(K/4)*8
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)

        # CNN Layer 3  (K/4)*(K/4)*8 ==> (K/4)*(K/4)*8
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)

        # squeezing
        x = torch.flatten(x,1)
        
        # FC Layer 1   (K/4)*(K/4)*8 ==> 64
        x = self.fc1(x)
        x = self.relu(x)

        # Output FC Layer 2  64 ==> out_ch
        x = self.fc2(x)

        return x