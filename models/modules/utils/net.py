import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),   #10*10*10
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)    #5*5*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),      #3*3*16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),       #1*1*32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1)
 
    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        y = self.conv3(y)
        # y = torch.reshape(y, [y.size(0), -1])
        y = self.conv4(y)
        # print(y)
        # print()
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        # print(category.shape)
        # print(offset.shape)
        # print("--------------------")
        return category, offset
    
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),   #22*22*28
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1)    #11*11*28
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),    #9*9*48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 0)          #4*4*48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1,
                 padding=0, dilation=1, groups=1),        #3*3*64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(3*3*64, 128)
        self.fc2 = nn.Linear(128, 5)
 
    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)
        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)
 
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset
    
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),   #46*46*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1)    #23*23*32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),    #21*21*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2)        #10*10*64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1),       #8*8*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2)            #4*4*64
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1,
                 padding=0, dilation=1, groups=1),  #3*3*128
            nn.BatchNorm2d(128),
            nn.ReLU()
         )
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 5)
 
    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = self.conv4(y)
        # print(y.shape,"==========")
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)
 
        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset
    
__all__ = ["PNet", "RNet", "ONet"]

if __name__ == "__main__":
    x = torch.randn((1, 3, 12, 12))
    pnet = PNet()
    rnet = RNet()