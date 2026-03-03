# ==============================
# CIFAR-10 分类训练脚本 (Resnet18 + TensorBoard)
# ==============================

# ------------------------------
# 1. 导入必要库
# ------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
import multiprocessing  # 新增：用于Windows多进程兼容

# ------------------------------
# 2. 全局配置
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU/CPU
EPOCHS = 100
BATCH_SIZE = 128

# ------------------------------
# 3. 数据预处理与加载
# ------------------------------
def get_dataloaders():
    """封装数据加载逻辑，避免多进程重复初始化"""
    # 定义训练集数据增强和标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 随机裁剪（带填充）
        transforms.RandomHorizontalFlip(),         # 随机水平翻转（50% 概率）
        transforms.ToTensor(),                     # 转为 PyTorch 张量 [0,1]
        transforms.Normalize(                      # 使用 CIFAR-10 官方均值/标准差
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # 测试集只做标准化（不增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # 加载 CIFAR-10 数据集（注意：root 应指向包含 cifar-10-batches-py/ 的目录）
    trainset = torchvision.datasets.CIFAR10(
        root='data/cifar10',      # 数据根目录
        train=True,               # 加载训练集
        download=False,           # 不下载（假设已存在）
        transform=train_transform # 应用训练变换
    )

    testset = torchvision.datasets.CIFAR10(
        root='data/cifar10',
        train=False,              # 加载测试集
        download=False,
        transform=test_transform  # 应用测试变换（无增强）
    )

    # 创建数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,             # 训练时打乱顺序
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False  # GPU 加速
    )

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,    # 测试也用大 batch 提速
        shuffle=False,            # 测试无需打乱
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return trainloader, testloader

# ------------------------------
# 4. 定义模型：resnet18
# ------------------------------
def get_model():
    """封装模型创建逻辑"""
    # 使用预训练模型（但不加载预训练权重）- 修复weights警告
    model = torchvision.models.resnet18(weights=None)  # 原weights=False改为weights=None，消除警告

    # 修改输入层（CIFAR-10 是 32x32，原 ResNet 为 224x224）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 修改输出层（10类）
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

# ------------------------------
# 5. 评估函数（计算测试准确率）
# ------------------------------
def evaluate(model, dataloader, device):
    """
    在测试集上评估模型准确率
    Args:
        model: 待评估模型
        dataloader: 测试数据加载器
        device: 设备（CPU/GPU）
    Returns:
        float: 准确率（0~1）
    """
    model.eval()  # 切换到评估模式（关闭 Dropout/BatchNorm 更新）
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ------------------------------
# 6. 主训练逻辑
# ------------------------------
def main():
    """主训练函数"""
    # 打印设备信息
    print(f"Using device: {DEVICE}")
    
    # 获取数据加载器
    trainloader, testloader = get_dataloaders()
    
    # 获取模型并移到指定设备
    model = get_model()
    model = model.to(DEVICE)
    
    # 初始化优化器、损失函数
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 适用于多分类任务
    
    # TensorBoard 日志记录器
    writer = SummaryWriter(log_dir='runs/cifar10_resnet18_gpu')

    # 训练循环
    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()  # 切换到训练模式
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            # 将数据移动到指定设备
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播 + 优化
            loss.backward()
            optimizer.step()
            
            # 累计损失（用于日志）
            running_loss += loss.item()
            
            # 每 100 个 batch 记录一次平均损失
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                global_step = epoch * len(trainloader) + batch_idx
                writer.add_scalar('Loss/train', avg_loss, global_step)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        # 每个 epoch 结束后评估测试集
        test_acc = evaluate(model, testloader, DEVICE)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Test Accuracy: {test_acc:.4f}")

    # 清理资源
    writer.close()
    print("训练完成！TensorBoard 日志已保存至 runs/cifar10_resnet18_gpu")
    print("运行以下命令查看结果：\n  tensorboard --logdir=runs")

# ------------------------------
# 7. Windows 多进程保护（核心修复）
# ------------------------------
if __name__ == '__main__':
    # Windows 多进程兼容必备
    multiprocessing.freeze_support()
    
    # 执行主训练逻辑
    main()