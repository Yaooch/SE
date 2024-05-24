import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=(2, 2))
        # BatchNorm层
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        # x的维度 : [batch_size, channels, time_frames, freq_bins]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

# 创建模型实例
model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'总参数数量为{total_params}')

x = torch.randn(1, 1, 100, 257)  # time_frames随不同的语音片段改变，这里取为100进行测试
x = x.to(device)
print(model(x).shape)