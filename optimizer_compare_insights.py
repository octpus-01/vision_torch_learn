# import everything
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter  # ← 新增：TensorBoard 支持
import os

# set matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# dataset prepare
x = torch.unsqueeze(torch.linspace(-1, 1, 500), dim=1)
y = x.pow(3)

# set parameters
LR = 0.01
batch_size = 15
epochs = 5
torch.manual_seed(10)

# load data
dataset = TensorDataset(x, y)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# 全局测试数据（用于画预测曲线）
test_x = x  # shape: (500, 1)
test_y = y  # shape: (500, 1)


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(n_input, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        x = torch.relu(self.hidden_layer(input))
        output = self.output_layer(x)
        return output


def train():
    net_SGD = Net(1, 10, 1)
    net_Momentum = Net(1, 10, 1)
    net_AdaGrad = Net(1, 10, 1)
    net_RMSprop = Net(1, 10, 1)
    net_Adam = Net(1, 10, 1)
    nets = [net_SGD, net_Momentum, net_AdaGrad, net_RMSprop, net_Adam]

    # optimizers
    optimizer_SGD = SGD(net_SGD.parameters(), lr=LR, momentum=0, weight_decay=0)
    optimizer_Momentum = SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
    optimizer_AdaGrad = Adagrad(net_AdaGrad.parameters(), lr=LR, weight_decay=0)
    optimizer_RMSprop = RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    optimizer_Adam = Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_AdaGrad, optimizer_RMSprop, optimizer_Adam]

    loss_func = nn.MSELoss()
    
    # 为每个优化器创建独立的 TensorBoard writer
    log_dirs = ['runs/SGD', 'runs/Momentum', 'runs/AdaGrad', 'runs/RMSprop', 'runs/Adam']
    writers = [SummaryWriter(log_dir) for log_dir in log_dirs]
    
    # 在第一个 epoch 开始前，向 TensorBoard 添加网络结构图（只需一次）
    dummy_input = torch.randn(1, 1)  # 模拟输入
    for i, net in enumerate(nets):
        writers[i].add_graph(net, dummy_input)

    step_count = 0  # 全局 step 计数器（用于 TensorBoard 横轴）

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
        
        # 训练阶段
        for step, (batch_x, batch_y) in enumerate(loader):
            for i, (net, optimizer, writer) in enumerate(zip(nets, optimizers, writers)):
                net.train()  # 设置为训练模式（虽然这里没 dropout/batchnorm，但好习惯）
                pred_y = net(batch_x)
                loss = loss_func(pred_y, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 记录 loss 到 TensorBoard
                writer.add_scalar('Loss/train', loss.item(), step_count)
            
            step_count += 1  # 每个 batch 算一步

        # ======== 每个 epoch 结束后：评估并画图 ========
        with torch.no_grad():  # 关闭梯度计算（节省内存）
            labels = ['SGD', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam']
            plt.figure(figsize=(12, 7))
            plt.plot(test_x.numpy(), test_y.numpy(), 'r-', label='真实函数 $y=x^3$', linewidth=2)
            
            avg_losses = []
            for i, (net, writer) in enumerate(zip(nets, writers)):
                net.eval()  # 设置为评估模式
                pred_test = net(test_x)
                epoch_loss = loss_func(pred_test, test_y).item()
                avg_losses.append(epoch_loss)
                
                # 记录 epoch loss 到 TensorBoard
                writer.add_scalar('Loss/epoch', epoch_loss, epoch)
                
                # 画预测曲线
                plt.plot(test_x.numpy(), pred_test.numpy(), '--', label=f'{labels[i]} (Loss={epoch_loss:.4f})')
            
            plt.legend(fontsize=10)
            plt.xlabel("x", size=12)
            plt.ylabel("y", size=12)
            plt.title(f"Epoch {epoch + 1} - 各优化器拟合效果", size=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 保存图片
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/epoch_{epoch+1:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()  # 释放内存
            
            # 打印当前 epoch 的平均损失
            print("Epoch Losses:")
            for name, loss_val in zip(labels, avg_losses):
                print(f"  {name}: {loss_val:.6f}")

    # 关闭所有 writer
    for writer in writers:
        writer.close()

    print("\n✅ 训练完成！")
    print("📊 查看 TensorBoard: 在终端运行 → tensorboard --logdir=runs")
    print("🖼️ 预测曲线图已保存到 ./plots/ 目录")


if __name__ == "__main__":
    train()