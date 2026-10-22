import torch
import torch.nn as nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)

        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avg_pool(x)

        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        print("Tensor after reshaping: ", x.shape)

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)

        return x


model = LeNet()

x = torch.randn(64,1, 32, 32)

ouput = model(x)

print(model)
summary(model, (1, 32, 32))
print("Output shape: ",ouput.shape)



# Output

"""  
Tensor after reshaping:  torch.Size([64, 120])
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv3): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
  (linear1): Linear(in_features=120, out_features=84, bias=True)
  (linear2): Linear(in_features=84, out_features=10, bias=True)
  (avg_pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (tanh): Tanh()
)
Tensor after reshaping:  torch.Size([2, 120])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             156
              Tanh-2            [-1, 6, 28, 28]               0
         AvgPool2d-3            [-1, 6, 14, 14]               0
            Conv2d-4           [-1, 16, 10, 10]           2,416
              Tanh-5           [-1, 16, 10, 10]               0
         AvgPool2d-6             [-1, 16, 5, 5]               0
            Conv2d-7            [-1, 120, 1, 1]          48,120
              Tanh-8            [-1, 120, 1, 1]               0
            Linear-9                   [-1, 84]          10,164
             Tanh-10                   [-1, 84]               0
           Linear-11                   [-1, 10]             850
================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.11
Params size (MB): 0.24
Estimated Total Size (MB): 0.35
----------------------------------------------------------------
Output shape:  torch.Size([64, 10])

"""
                



