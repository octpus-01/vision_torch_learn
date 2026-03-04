# ==============================
# CIFAR-10 分类训练脚本 (Resnet18 + TensorBoard)
# 最终完美版：无重复打印 + TensorBoard正常显示
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
import os

# ------------------------------
# 2. 全局配置（仅定义，不执行打印）
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BASE_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 4  # 等效batch size=256（不是1024，你之前写错了）
NUM_WORKERS = 2
PIN_MEMORY = True
MIXED_PRECISION = True

# ------------------------------
# 3. 工具函数（避免全局执行）
# ------------------------------
def print_system_info():
    """仅在main函数中执行一次系统信息打印"""
    print("="*60)
    print("系统/设备信息")
    print("="*60)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Effective Batch Size: {BASE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60 + "\n")

# ------------------------------
# 4. 数据预处理
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

    # 数据加载器
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
# 5. 模型定义
# ------------------------------
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model

# ------------------------------
# 6. 数据传输函数
# ------------------------------
def move_to_device(batch, device):
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

# ------------------------------
# 7. 评估函数
# ------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
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
# 8. 主训练逻辑
# ------------------------------
def main():
    # 1. 只打印一次系统信息（核心修复：解决重复打印）
    print_system_info()
    
    # 2. GPU内存优化
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
    
    # 3. 获取数据和模型
    trainloader, testloader = get_dataloaders()
    model = get_model()
    
    # 4. 初始化混合精度缩放器
    scaler = None
    if MIXED_PRECISION and torch.cuda.is_available():
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    
    # 5. 优化器
    optimizer_kwargs = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
    if torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused'):
        optimizer_kwargs['fused'] = True
    
    optimizer = Adam(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # 6. TensorBoard配置（核心修复：确保日志写入正常）
    log_dir = 'runs/cifar10_resnet18_260304-2'
    # 清空旧日志（避免缓存问题）
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 7. 训练初始化
    print("开始训练...\n")
    global_step = 0  # 确保step连续递增
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(trainloader):
            inputs, labels = move_to_device(batch, DEVICE)
            
            # 混合精度前向传播
            if scaler is not None:
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss_step = loss / GRADIENT_ACCUMULATION_STEPS
                except ImportError:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss_step = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_step = loss / GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss_step).backward()
            else:
                loss_step.backward()
            
            # 梯度累积更新
            batch_count += 1
            if batch_count % GRADIENT_ACCUMULATION_STEPS == 0:
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
                
                optimizer.zero_grad(set_to_none=True)
                
                # 累加损失（核心修复：使用原始loss值）
                running_loss += loss.item()
                batch_count = 0  # 重置累积计数
                
                # 每10个更新步骤记录一次日志（确保step连续）
                if global_step % 10 == 0:
                    avg_loss = running_loss / 10
                    # 写入TensorBoard（核心：global_step每次递增）
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    writer.add_scalar('Loss/epoch', avg_loss, epoch)
                    
                    # 打印日志
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3
                        print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(trainloader)}], "
                              f"Step: {global_step}, Loss: {avg_loss:.4f}, GPU Mem: {gpu_mem:.2f} GB")
                    else:
                        print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(trainloader)}], "
                              f"Step: {global_step}, Loss: {avg_loss:.4f}")
                    
                    running_loss = 0.0
            
            global_step += 1  # 关键：每次batch都递增，确保step连续
        
        # 每个epoch评估并记录准确率
        test_acc = evaluate(model, testloader, DEVICE)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Time/epoch', time.time() - epoch_start, epoch)
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Complete - Time: {time.time() - epoch_start:.2f}s, Test Accuracy: {test_acc:.4f}\n")
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc
            }, f'{log_dir}/cifar10_resnet18_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}\n")

    # 最终保存
    torch.save(model.state_dict(), f'{log_dir}/cifar10_resnet18_final.pth')
    writer.close()
    
    # 打印TensorBoard启动命令
    print("="*60)
    print("训练完成！")
    print(f"最终模型保存至: {log_dir}/cifar10_resnet18_final.pth")
    print("\n启动TensorBoard查看日志：")
    print(f"tensorboard --logdir={log_dir}")
    print("="*60)

# ------------------------------
# 9. 主函数入口
# ------------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()