### 🔹 第一部分：导入必要的库

```python
import torch
```
> 导入 PyTorch —— 这是做深度学习的核心工具，用来处理张量（多维数组）和构建神经网络。

```python
import torch.nn as nn
```
> `nn` 是 PyTorch 的“神经网络模块”，里面有很多现成的层（比如全连接层、激活函数等），不用自己从头写。

```python
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
```
> - `TensorDataset`：把输入 `x` 和输出 `y` 打包成一个数据集。
> - `DataLoader`：负责按批次（batch）加载数据，还能打乱顺序、多线程加载等。

```python
from torch.optim.adam import Adam
```
> 导入 **Adam 优化器** —— 一种聪明的“调参工具”，能自动调整神经网络的参数，让模型学得更快更好。

```python
import numpy as np
```
> NumPy 是处理数值计算的常用库，比如生成数组、数学函数（如 `cos`）。

```python
import matplotlib
import matplotlib.pyplot as plt
```
> 用来画图的库。`plt` 就是画图的主要工具。

---

### 🔹 第二部分：设置中文字体（避免中文显示为方框）

```python
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False
```
> 让图表中的中文（比如坐标轴标签）能正常显示，而不是乱码或方块。  
> （如果你电脑没装这个字体，可能会报错，可以注释掉这两行）

---

### 🔹 第三部分：准备训练数据（目标函数是 cos(x)）

```python
x = np.linspace(-2*np.pi, 2*np.pi, 400)
```
> 在区间 $[-2\pi, 2\pi]$ 内**均匀取 400 个点**作为输入 `x`。  
> 比如：-6.28, -6.25, ..., 0, ..., 6.25, 6.28。

```python
y = np.cos(x)
```
> 对每个 `x`，计算 `y = cos(x)`，这就是我们想让神经网络学会的“真实规律”。

```python
X = np.expand_dims(x, axis=1)
```
> 把一维的 `x`（形状 `(400,)`）变成二维（形状 `(400, 1)`），即**每行一个样本，每列一个特征**。  
> 这是因为神经网络要求输入是“表格形式”（就像 Excel 表格）。

```python
Y = y.reshape(400, -1)
```
> 同样把 `y` 变成二维（`(400, 1)`）。`-1` 表示“自动算出这一维的大小”，这里就是 1。
> 与Y = np.expand_dims(y, axis=1)等价

---

### 🔹 第四部分：把数据转成 PyTorch 能用的格式，并打包

```python
dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))
```
> - `torch.tensor(...)`：把 NumPy 数组转成 PyTorch 张量（并指定数据类型为浮点数）。
> - `TensorDataset`：把 `X` 和 `Y` 配对，形成一个“数据集”，之后可以像列表一样遍历 `(x_i, y_i)`。

```python
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```
> - `batch_size=10`：每次训练拿 10 个数据点（称为一个“批次”）。
> - `shuffle=True`：每次遍历数据前都**打乱顺序**，防止模型“记住顺序”而过拟合。

---

### 🔹 第五部分：定义神经网络模型

```python
class Net(nn.Module):
```
> 定义一个叫 `Net` 的神经网络类，继承自 `nn.Module`（PyTorch 的标准做法）。

```python
    def __init__(self):
        super(Net, self).__init__()
```
> 初始化父类（必须写），然后开始搭建网络结构。

```python
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
```
> 用 `Sequential` 把几层网络串起来：
> - 第1层：输入1个数（x），输出10个数 → 然后用 ReLU 激活（把负数变0）
> - 第2层：10 → 100 → ReLU
> - 第3层：100 → 10 → ReLU
> - 第4层：10 → 1（最终输出 y）
>
> ✅ 这是一个**全连接神经网络**（也叫多层感知机），用来拟合 `cos(x)` 曲线。

```python
    def forward(self, input: torch.FloatTensor):
        return self.net(input)
```
> 定义“前向传播”：当输入一个 `x`，就让它依次通过上面定义的网络，得到预测值 `y`。

---

### 🔹 第六部分：创建模型、优化器和损失函数

```python
net = Net()
```
> 创建一个 `Net` 的实例，也就是真正的神经网络对象。

```python
optim = Adam(net.parameters(), lr=0.001)
```
> - `net.parameters()`：获取神经网络里所有可训练的参数（权重和偏置）。
> - `lr=0.001`：学习率，控制每次更新参数的步长（太大会跳过最优解，太小会学得太慢）。

```python
Loss = nn.MSELoss()
```
> 使用**均方误差（MSE）** 作为损失函数。  
> 它衡量“预测值”和“真实值”之间的平均平方差距，越小越好。

---

### 🔹 第七部分：训练模型（100 轮）

```python
for epoch in range(100):
```
> 训练 100 轮（epoch），每轮遍历全部数据一次。

```python
    loss = None
```
> 初始化 `loss`，避免后面打印时报错。

```python
    for batch_x, batch_y in dataloader:
```
> 从 `dataloader` 中每次取出一个批次（10 个 `(x, y)` 对）。

```python
        y_predict = net(batch_x)
```
> 把 `batch_x` 输入网络，得到预测值 `y_predict`。

```python
        loss = Loss(y_predict, batch_y)
```
> 计算当前批次的损失（预测 vs 真实）。

```python
        optim.zero_grad()
```
> 清空上一次的梯度（否则会累加，导致错误）。

```python
        loss.backward()
```
> **反向传播**：自动计算每个参数对损失的影响（即“梯度”）。

```python
        optim.step()
``>
> **更新参数**：根据梯度和优化器规则，微调神经网络的权重，让下次预测更准。

```python
    if (epoch+1) % 10 == 0:
        if loss is not None:
            print("训练步骤：{0}，模型损失{1}".format(epoch+1, loss.item()))
```
> 每训练 10 轮，打印一次当前的损失值，看看模型是不是在进步（损失应该越来越小）。

---

### 🔹 第八部分：用训练好的模型做预测并画图

```python
predict = net(torch.tensor(X, dtype=torch.float))
```
> 用整个 `X`（400 个点）输入训练好的模型，得到预测结果 `predict`。

```python
plt.figure(figsize=(12,7), dpi=160)
```
> 创建一个大一点、清晰一点的画布。

```python
plt.plot(x, y, label="real", marker="X")
```
> 画出真实的 `cos(x)` 曲线，用 "X" 标记数据点。

```python
plt.plot(x, predict.detach().numpy(), label="predict", marker='o')
```
> - `predict.detach()`：从 PyTorch 的“自动求导系统”中分离出来（因为我们只是要数值，不要梯度）。
> - `.numpy()`：转成 NumPy 数组，方便画图。
> - 用 "o" 标记预测点。

```python
plt.xlabel("x", size=15)
plt.ylabel("cos(x)", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(fontsize=15)
```
> 设置坐标轴标签、刻度字号、图例大小，让图更美观。

```python
plt.savefig('cos.png')
```
> 把图画保存为 `cos.png` 文件（在当前目录下）。

---

### ✅ 总结：这段代码在做什么？

> **用一个小型神经网络去学习 `y = cos(x)` 的函数关系**，  
> 通过 100 轮训练，让它能根据任意 `x` 预测出接近 `cos(x)` 的值，  
> 最后画出“真实值”和“预测值”的对比图，验证效果。

---

