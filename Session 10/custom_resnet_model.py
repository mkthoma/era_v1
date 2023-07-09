import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Model
class custom_ResNet(nn.Module):
    def __init__(self):
        super(custom_ResNet, self).__init__()


        # PREPARATION BLOCK
        self.prepblock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            nn.ReLU(),nn.BatchNorm2d(64))
            # output_size = 32, RF=3


        # CONVOLUTION BLOCK 1
        self.convblock1_l1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            # output_size = 32, RF=5
            nn.MaxPool2d(2, 2),nn.ReLU(),nn.BatchNorm2d(128))
            # output_size = 16, RF=6

        self.convblock1_r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            nn.ReLU(),nn.BatchNorm2d(128),
            # output_size = 16, RF=10
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            nn.ReLU(),nn.BatchNorm2d(128))
            # output_size = 16, RF=14


        # CONVOLUTION BLOCK 2
        self.convblock2_l1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            # output_size = 16, RF=18
            nn.MaxPool2d(2, 2),nn.ReLU(),nn.BatchNorm2d(256))
            # output_size = 8, RF=20


        # CONVOLUTION BLOCK 3
        self.convblock3_l1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            # output_size = 8, RF=28
            nn.MaxPool2d(2, 2),
            nn.ReLU(),nn.BatchNorm2d(512))
            # output_size = 4, RF=32


        self.convblock3_r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            nn.ReLU(),nn.BatchNorm2d(512),
             # output_size = 4, RF=48
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
            nn.ReLU(),nn.BatchNorm2d(512))
            # output_size = 4, RF=64


        # CONVOLUTION BLOCK 4
        self.convblock4_mp = nn.Sequential(nn.MaxPool2d(4))
        # output_size = 1, RF = 88


        # OUTPUT BLOCK - Fully Connected layer
        self.output_block = nn.Sequential(nn.Linear(in_features=512, out_features=10, bias=False))
        # output_size = 1, RF = 88


    def forward(self, x):

        # Preparation Block
        x1 = self.prepblock(x)

        # Convolution Block 1
        x2 = self.convblock1_l1(x1)
        x3 = self.convblock1_r1(x2)
        x4 = x2 + x3

        # Convolution Block 2
        x5 = self.convblock2_l1(x4)

        # Convolution Block 3
        x6 = self.convblock3_l1(x5)
        x7 = self.convblock3_r2(x6)
        x8 = x7 + x6

        # Convolution Block 4
        x9 = self.convblock4_mp(x8)

        # Output Block
        x9 = x9.view(x9.size(0), -1)
        x10 = self.output_block(x9)
        return F.log_softmax(x10, dim=1)

    # Model Summary
    def model_summary(self, input_size):
        return summary(self, input_size)


