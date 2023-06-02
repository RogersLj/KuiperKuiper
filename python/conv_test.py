import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    
def main():
    x = torch.randn(1, 3, 32, 32)
    c = Conv(3, 3, 3)
    y = c(x)
    print(y.shape)