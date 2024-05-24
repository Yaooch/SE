from torch.utils.data import Dataset, DataLoader
from torch import nn
import time
from torch.optim.lr_scheduler import StepLR
from dataset import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on:', device)

method_list = ['ibm', 'iam', 'irm', 'psm', 'orm']

for method in method_list:
    print(f'——————————当前训练的目标为{method}——————————')
    train_dataset = GetDataset(method=method)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CNN()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数数量为{total_params}')

    num_epochs = 100
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    loss = nn.MSELoss()
    loss.to(device)
    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0
        for data in train_loader:
            feature, label = data['feature'], data['label']
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            loss_value = loss(output, label)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()

        # 更新学习率
        scheduler.step()
        end_time = time.time()
        print(f'第{epoch + 1}轮训练结束，Loss: {running_loss / len(train_dataset)}，用时{end_time - start_time}秒')

    print(f'训练轮数为{num_epochs}，学习率为{optimizer.param_groups[0]["lr"]}. ')
    torch.save(model.state_dict(), '../model/model_' + method + '.pth')