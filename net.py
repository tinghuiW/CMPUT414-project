# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:11:50 2019

@author: 60418
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# This net is for audio-stream #

class asNet(nn.Module):
    
    def __init__(self):
        
        super(asNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 96, kernel_size = (1,7), dilation = (1,1), padding = (0,3))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (7,1), dilation = (1,1), padding = (3,0))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (1,1), padding = (2,2))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (2,1), padding = (4,2))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (4,1), padding = (8,2))
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (8,1), padding = (16,2))
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (16,1), padding = (32,2))
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = (32,1), padding = (64,2))
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 1, padding = 2)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 2, padding = 4)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 4, padding = 8)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 8, padding = 16)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 16, padding = 32)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (5,5), dilation = 32, padding = 64)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(in_channels = 96, out_channels = 8, kernel_size = (5,5), dilation = 1, padding = 2)
        self.relu15 = nn.ReLU()
        #self.fl = torch.flatten()
        self.relu16 = nn.ReLU()
       
        
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
        layer7 = self.conv7(layer6)
        layer7 = self.relu7(layer7)
        layer8 = self.conv8(layer7)
        layer8 = self.relu8(layer8)
        layer9 = self.conv9(layer8)
        layer9 = self.relu9(layer9)
        layer10 = self.conv10(layer9)
        layer10 = self.relu10(layer10)
        layer11 = self.conv11(layer10)
        layer11 = self.relu11(layer11)
        layer12 = self.conv12(layer11)
        layer12 = self.relu12(layer12)
        layer13 = self.conv13(layer12)
        layer13 = self.relu13(layer13)
        layer14 = self.conv14(layer13)
        layer14 = self.relu14(layer14)
        layer15 = self.conv15(layer14)
        layer15 = self.relu15(layer15)
        flattenLayer = torch.flatten(layer15)
        flattenLayer = self.relu7(flattenLayer)
        
        
        out1 = flattenLayer
        
        return out1


# This net is video-stream #
class vsNet(nn.Module):
    
    def __init__(self):
        
        super(vsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = (7,1), dilation = (1,1), padding = (3,0))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (1,1), padding = (2, 0))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (2,1), padding = (4,0))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (4,1), padding = (8,0))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (8,1), padding = (16,0))
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (5,1), dilation = (16,1), padding = (32,0))
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
    


# test #
#net = vsNet()
#input = Variable(torch.zeros(256, 1024, 75, 1))
#out = net(input)
#print(out)














