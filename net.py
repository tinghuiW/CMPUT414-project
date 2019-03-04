# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:11:50 2019

@author: 60418
"""

import torch
import torch.nn as nn

# This net is for audio-stream #

class asNet(nn.Module):
    
    def __init__(self):
        
        super(asNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = (1,7), dilation = (1,1))
        self.relu1 = nn.ReLu()
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (7,1), dilation = (1,1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (1,1))
        self.relu3 = nn.ReLu()
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (2,1))
        self.relu4 = nn.ReLu()
        self.conv5 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (4,1))
        self.relu5 = nn.ReLu()
        self.conv6 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (8,1))
        self.relu6 = nn.ReLu()
        # it is not finished #
        
    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.relu1(layer1)
        layer2 = self.conv2(layer1)
        layer2 = self.relu2(layer2)
        layer3 = self.conv3(layer2)
        layer3 = self.relu3(layer3)
        layer4 = self.conv4(layer3)
        layer4 = self.relu4(layer4)
        layer5 = self.conv5(layer4)
        layer5 = self.relu5(layer5)
        layer6 = self.conv6(layer5)
        layer6 = self.relu6(layer6)
        # it is not finshed #
        
        out1 = layer6
        
        return out1


class vsNet(nn.Module):
    
    def __init__(self):
        
        super(vsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (7,1), dilation = (1,1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (1,1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (2,1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (4,1))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (8,1))
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (16,1))
        self.relu6 = nn.ReLU()
        
    
    def forward(self,x):
        layer1 = self.conv1(x)
        layer1 = self.relu1(layer1)
        layer2 = self.conv2(layer1)
        layer2 = self.relu2(layer2)
        layer3 = self.conv3(layer2)
        layer3 = self.relu3(layer3)
        layer4 = self.conv4(layer3)
        layer4 = self.relu4(layer4)
        layer5 = self.conv5(layer4)
        layer5 = self.relu5(layer5)
        layer6 = self.conv6(layer5)
        layer6 = self.relu6(layer6)
        
        out2 = layer6
        
        return out2 




































