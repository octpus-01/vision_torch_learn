# import everything
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter

# presets
epochs = 100
DEVICE = torch.device("cpu")


# 1. 定义数据转换（关键！）
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 数据增强
    transforms.RandomHorizontalFlip(),     # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 标准化
])

# 2. 加载数据集（注意 root 路径！）
trainset = torchvision.datasets.CIFAR10(
    root='data/cifar10',  # 你的数据目录
    train=True,
    download=False,       # 已存在，不下载
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='data/cifar10',
    train=False,
    download=False,
    transform=transform
)

# 3. 创建 DataLoader（关键：batch_size 和 shuffle）
trainloader = DataLoader(
    trainset,
    batch_size=128,       # 常用值
    shuffle=True,         # 打乱数据
    num_workers=2         # 多进程加载
)

testloader = DataLoader(
    testset,
    batch_size=1,
    shuffle= True,
    num_workers=2
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3通道输入，32个卷积核
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # 64通道 * 8x8特征图
            nn.ReLU(),
            nn.Linear(128, 10)            # 10类输出
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.classifier(x)


# 1. 选择模型
model = SimpleCNN().to(DEVICE)

# 2. 定义优化器（Adam 适合图像任务）
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 1. 初始化 writer（日志目录）
writer = SummaryWriter(log_dir='runs/cifar10_cnn')

def evaluate(model, dataloader):
    model.eval()  # 评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 2. 在训练循环中记录

for epoch in range(epochs):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # ... 前向传播、计算损失、反向传播 ...
         
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 记录训练损失
        writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + batch_idx)
        
        # 记录学习率
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch * len(trainloader) + batch_idx)


    # 每个 epoch 结束，记录测试准确率
    test_acc = evaluate(model, testloader)
    writer.add_scalar('Accuracy/test', test_acc, epoch)



