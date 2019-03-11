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
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 96, kernel_size = (1,7), dilation = (1,1))
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


# This net is video-stream #
class vsNet(nn.Module):
    
    def __init__(self):
        
        super(vsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = (7,1), dilation = (1,1))
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
#        self.fl = torch.flatten()
#        self.relu7 = nn.ReLU()
        
    
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
#        flattenLayer = self.fl(layer6)
#        flattenLayer = self.relu7(flattenLayer)
        
        out2 = layer6
        
        return out2 


# fusion net #
class avNet(nn.modules):
    
    def __init__(self, net1, net2):
        
        super(avNet, self).__init__()
        self.fusionLayer = torch.cat((net1, net2), 0)
        self.fuRelu = nn.ReLU()
        self.biLSTMLayer = nn.LSTM(2568, hidden_size = 400, num_layers = 1, bidirectional=True)
        self.biLSTMRelu = nn.ReLU()
        self.fcLayer1 = nn.Linear(400, 600)
        self.fcRelu1 = nn.ReLU()
        self.fcLayer2 = nn.Linear(600, 600)
        self.fcRelu2 = nn.ReLU()
        self.fcLayer3 = nn.Linear(600, 600)
        self.fcSig = torch.sigmoid()
        
        
    def forward(self, x, ori):
        fsLayer = self.fusionLayer(x)
        fsRule = self.fuRelu(fsLayer)
        biLayer = self.biLSTMLayer(fsRule)
        biRule = self.biLSTMRelu(biLayer)
        layer1 = self.fcLayer1(biRule)
        layer1 = self.fcRelu1(layer1)
        layer2 = self.fcLayer2(layer1)
        layer2 = self.fcRelu2(layer2)
        layer3 = self.fcLayer3(layer2)
        layer3 = self.fcSig(layer3)
        
        # complex mask #
        out3 = layer3
        out3 = torch.reshape(out3, shape = (-1, 298, 257*2))
        
        sound1 = out3[:, :, :257] * ori[:, 0, :, :]
        sound2 = out3[:, :, 257:] * ori[:, 0, :, :]
        sound = torch.cat((sound1, sound2), 1)
        
        return sound
    































