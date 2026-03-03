# ==============================
# CIFAR-10 分类训练脚本 (Resnet18 + TensorBoard)
# 最终稳定版：无依赖 + 低内存 + 高GPU利用率
# ==============================

# ------------------------------
# 1. 导入必要库
# ------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import time

# ------------------------------
# 2. 全局配置（稳定版）
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BASE_BATCH_SIZE = 64  # 适配8GB GPU内存
GRADIENT_ACCUMULATION_STEPS = 4  # 等效batch size=256
NUM_WORKERS = 0  # 单进程避免页面文件不足
PIN_MEMORY = True
MIXED_PRECISION = True

# 打印设备信息（仅基础信息，无依赖）
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Effective Batch Size: {BASE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"PyTorch Version: {torch.__version__}")

# ------------------------------
# 3. 数据预处理
# ------------------------------
def get_dataloaders():
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='data/cifar10',
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='data/cifar10',
        train=False,
        download=True,
        transform=test_transform
    )

    # 数据加载器（单进程稳定版）
    trainloader = DataLoader(
        trainset,
        batch_size=BASE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    testloader = DataLoader(
        testset,
        batch_size=BASE_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return trainloader, testloader

# ------------------------------
# 4. 模型定义
# ------------------------------
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    model = model.to(DEVICE)
    # 多GPU支持（如有）
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model

# ------------------------------
# 5. 数据传输函数
# ------------------------------
def move_to_device(batch, device):
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

# ------------------------------
# 6. 评估函数
# ------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    # 混合精度评估
    if MIXED_PRECISION and torch.cuda.is_available():
        try:
            from torch.amp import autocast
            with autocast('cuda'):
                for batch in dataloader:
                    inputs, labels = move_to_device(batch, device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        except ImportError:
            from torch.cuda.amp import autocast
            with autocast():
                for batch in dataloader:
                    inputs, labels = move_to_device(batch, device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    else:
        for batch in dataloader:
            inputs, labels = move_to_device(batch, device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"Evaluation Time: {time.time() - start_time:.2f}s, Test Accuracy: {acc:.4f}")
    return acc

# ------------------------------
# 7. 主训练逻辑
# ------------------------------
def main():
    # 初始化混合精度缩放器
    scaler = None
    if MIXED_PRECISION and torch.cuda.is_available():
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    
    # GPU内存优化
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
    
    # 获取数据和模型
    trainloader, testloader = get_dataloaders()
    model = get_model()
    
    # 优化器
    optimizer_kwargs = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    # 兼容fused Adam
    if torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused'):
        optimizer_kwargs['fused'] = True
    
    optimizer = Adam(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    writer = SummaryWriter(log_dir='runs/cifar10_resnet18_stable')

    print("\n开始训练（稳定版）...")
    global_step = 0
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(trainloader):
            inputs, labels = move_to_device(batch, DEVICE)
            
            # 混合精度前向传播
            if scaler is not None:
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                except ImportError:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # 高效梯度清零
                optimizer.zero_grad(set_to_none=True)
                
                # 日志记录（移除GPU利用率查询）
                running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    
                    # 仅打印GPU内存（无额外依赖）
                    if torch.cuda.is_available():
                        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3
                        print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(trainloader)}], "
                              f"Loss: {avg_loss:.4f}, GPU Mem Used: {gpu_mem_used:.2f} GB, Cached: {gpu_mem_cached:.2f} GB")
                    else:
                        print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {avg_loss:.4f}")
                    
                    running_loss = 0.0
                    global_step += 1
        
        # 每个epoch评估一次
        test_acc = evaluate(model, testloader, DEVICE)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Complete - Time: {time.time() - epoch_start:.2f}s, Test Accuracy: {test_acc:.4f}\n")
        
        # 保存模型（每10个epoch保存一次，减少IO）
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc
            }, f'runs/cifar10_resnet18_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}\n")

    # 训练结束保存最终模型
    torch.save(model.state_dict(), 'runs/cifar10_resnet18_final.pth')
    writer.close()
    print("训练全部完成！最终模型已保存至 runs/cifar10_resnet18_final.pth")

# ------------------------------
# 8. 主函数入口
# ------------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()