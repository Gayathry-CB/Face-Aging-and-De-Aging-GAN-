import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
   def __init__(self, channels):
       super(ResidualBlock, self).__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
           nn.InstanceNorm2d(channels),
           nn.ReLU(inplace=True),
           nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
           nn.InstanceNorm2d(channels),
       )


   def forward(self, x):
       return x + self.conv(x)


class Generator(nn.Module):
   def __init__(self, input_channels=3, output_channels=3, num_residuals=6):
       super(Generator, self).__init__()
       self.initial = nn.Sequential(
           nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
           nn.InstanceNorm2d(64),
           nn.ReLU(inplace=True)
       )


       self.res_blocks = nn.Sequential(
           *[ResidualBlock(64) for _ in range(num_residuals)]
       )


       self.output = nn.Sequential(
           nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3, bias=False),
           nn.Tanh()
       )


   def forward(self, x):
       x = self.initial(x)
       x = self.res_blocks(x)
       return self.output(x)
