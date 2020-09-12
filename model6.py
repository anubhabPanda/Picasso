import torch
import torch.nn as nn
import torch.nn.functional as F



class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)




class Net(nn.Module):
  def __init__(self, batch_normalization_type="BN" , gbn_splits=0):
    super(Net, self).__init__() 
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # Input=28x28x1 Kernel=3x3x1x10 Output=28x28x10 RF=3x3
        nn.ReLU(),
        nn.BatchNorm2d(10) if batch_normalization_type=="BN" else GhostBatchNorm(10, gbn_splits),
        nn.Dropout(0.04),
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # Input=28x28x10 Kernel=3x3x10x10 Output=28x28x10 RF=5x5
        nn.ReLU(),
        nn.BatchNorm2d(10) if batch_normalization_type=="BN" else GhostBatchNorm(10, gbn_splits),
        nn.Dropout(0.04)
        ) #output_size=28x28x10
    self.conv2 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2), # Input=28x28x10 Output=14x14x10 RF=6x6
        )#output_size=14x14x10

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), #Input=14x14x10 Kernel=3x3x10x10 Output=12x12x10 RF=10x10
        nn.ReLU(),
        nn.BatchNorm2d(10) if batch_normalization_type=="BN" else GhostBatchNorm(10, gbn_splits),
        nn.Dropout(0.04),
        nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Input=12x12x10 Kernel=3x3x10x12 Output=10x10x12 RF=14x14
        nn.ReLU(),
        nn.BatchNorm2d(12) if batch_normalization_type=="BN" else GhostBatchNorm(12, gbn_splits),
        nn.Dropout(0.04),
        nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), #Input=10x10x12 Kernel=3x3x12x16 Output=8x8x16 RF=18x18
        nn.ReLU(),            
        nn.BatchNorm2d(16) if batch_normalization_type=="BN" else GhostBatchNorm(16, gbn_splits),
        nn.Dropout(0.04),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), #Input=8x8x16 Kernel=3x3x16x16 Output=6x6x16 RF=22x22
        nn.ReLU(),         
        nn.BatchNorm2d(16) if batch_normalization_type=="BN" else GhostBatchNorm(16, gbn_splits),
        nn.Dropout(0.04)
        )#output_size=6x6x16

    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), #Input=6x6x16 Kernel=3x3x16x16 Output=4x4x16 RF=26x26
        nn.ReLU(),
        nn.BatchNorm2d(16) if batch_normalization_type=="BN" else GhostBatchNorm(16, gbn_splits),
        nn.AvgPool2d(kernel_size=(4,4)), #Input=4x4x16 Kernel=4x4x16x16 Output=1x1x16
        nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False) #Input=1x1x16 Kernel=1x1x16x10 Output=1x1x10
        ) 


  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=1)