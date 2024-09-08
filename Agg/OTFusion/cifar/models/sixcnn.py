from torch import nn
import numpy as np
import torch


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


# class CifarModel(nn.Module):
#     def __init__(self, inputsize):
#         super().__init__()
#
#         ncha, size, _ = inputsize
#         self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(size, 3, padding=1)  # 32
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(s, 3, padding=1)  # 32
#         s = s // 2  # 16
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(s, 3, padding=1)  # 16
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(s, 3, padding=1)  # 16
#         s = s // 2  # 8
#         self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(s, 3, padding=1)  # 8
#         self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
#         s = compute_conv_output_size(s, 3, padding=1)  # 8
#         #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         #         s = compute_conv_output_size(s,3, padding=1) # 8
#         s = s // 2  # 4
#         self.fc1 = nn.Linear(s * s * 128, 1024, bias=False)  # 2048
#         self.drop1 = nn.Dropout(0.25)
#         self.drop2 = nn.Dropout(0.5)
#         self.MaxPool = torch.nn.MaxPool2d(2)
#         self.avg_neg = []
#         # self.fc2 = nn.Linear(256, 100)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x, avg_act=False):
#         if x.size(1) != 3:
#             bsz = x.size(0)
#             x = x.view(bsz, 3, 32, 32)
#         act1 = self.relu(self.conv1(x))
#         act2 = self.relu(self.conv2(act1))
#         h = self.drop1(self.MaxPool(act2))
#         act3 = self.relu(self.conv3(h))
#         act4 = self.relu(self.conv4(act3))
#         h = self.drop1(self.MaxPool(act4))
#         act5 = self.relu(self.conv5(h))
#         act6 = self.relu(self.conv6(act5))
#         h = self.drop1(self.MaxPool(act6))
#         h = h.view(x.shape[0], -1)
#         act7 = self.relu(self.fc1(h))
#         h = self.drop2(act7)
#         return h


class SixCNN(nn.Module):
    def __init__(self, inputsize, outputsize=100,nc_per_task = 10):
        super().__init__()
        self.outputsize = outputsize
        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(size, 3, padding=1)  # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = nn.Linear(s * s * 128, 1024, bias=False)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
        self.last = nn.Linear(1024, outputsize, bias=False)
        self.nc_per_task = nc_per_task
    def forward(self, x, t=-1, pre=False, is_cifar=True, avg_act=False):
        if x.size(1) != 3:
            bsz = x.size(0)
            x = x.view(bsz, 3, 32, 32)
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        output = self.last(h)
        if is_cifar and t != -1:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.outputsize:
                output[:, offset2:self.outputsize].data.fill_(-10e10)
        return output
