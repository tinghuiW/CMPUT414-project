#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:00:13 2019

@author: WTH
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# This net is for audio-stream #


class asNet(nn.Module):

    def __init__(self):

        super(asNet, self).__init__()
        # using Sequential to simplfy the forword part #
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=(1, 7), dilation=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 1), dilation=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(2, 1), padding=(4, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(4, 1), padding=(8, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(8, 1), padding=(16, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(16, 1), padding=(32, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(32, 1), padding=(64, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=1, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=2, padding=4),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=4, padding=8),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=8, padding=16),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=16, padding=32),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=32, padding=64),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=(5, 5), dilation=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )


    def forward(self, x):
        out = self.net(x)    # [batch, 8, 297, 257]
        out = torch.transpose(out, 1, 2)    # [batch , 297, 8, 257] Adjust size
        out = torch.reshape(out, [-1, 297, 8 * 257])    # [bath, 297, 8*257] making out size correct

        return out


# This net is video-stream #
class vsNet(nn.Module):

    def __init__(self):

        super(vsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(7, 1), dilation=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False, dilation=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(4, 0), bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(8, 0), bias=False, dilation=(4, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(16, 0), bias=False, dilation=(8, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(5, 1), stride=1, padding=(32, 0), bias=False, dilation=(16, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.UpsamplingNearest2d((297, 1)), # upsampling the layer, let it be the correct output size
        )


    def forward(self, x):
        out = self.net(x)    # [batch, 256, 297, 1]
        out = torch.transpose(out, 1, 2)    # [batch, 297, 256, 1]
        out = torch.reshape(out, [-1, 297, 256])    # [batch, 297, 256] making output size same as the paper gives

        return out


# fusion net #
class avNet(nn.Module):

    def __init__(self):

        super(avNet, self).__init__()
        self.asnet = asNet()
        self.vsnet = vsNet()

        self.biLSTMLayer = nn.LSTM(256 * 2 + 8 * 257, hidden_size=400, num_layers=1, bidirectional=True, batch_first=True)
        self.fcLayer1 = nn.Linear(800, 600)
        self.fcRelu1 = nn.ReLU()
        self.fcLayer2 = nn.Linear(600, 600)
        self.fcRelu2 = nn.ReLU()
        self.fcLayer3 = nn.Linear(600, 600)


        self.fcMask= nn.Linear(600, 257*2*2)

    def forward(self, video_1, video_2, audio):
        video_1_feature = self.vsnet(video_1) # first video net
        video_2_feature = self.vsnet(video_2) # second video net
        audio_feature = self.asnet(audio) # audio net

        fuse_feature = torch.cat([video_1_feature, video_2_feature, audio_feature], dim=2) # connect three net.

        out, _ = self.biLSTMLayer(fuse_feature) #[batch, 297, 800]
        out = self.fcLayer1(out)
        out = self.fcRelu1(out)
        out = torch.sigmoid(self.fcMask(out))
        out = torch.reshape(out, [-1, 297, 257,2]) # [batch , 297, 247 ,2] # adjust the output size


        return out


if __name__ == "__main__":
    device = torch.device('cuda')

    # test input
    test_video_1_input = torch.randn(1, 1024, 75, 1).to(device)
    test_video_2_input = torch.randn(1, 1024, 75, 1).to(device)
    test_audio_input = torch.randn(1, 2, 297, 257).to(device)

    # forward pass
    avnet = avNet().to(device)
    out = avnet(test_video_1_input, test_video_2_input, test_audio_input)

    print(out.shape)