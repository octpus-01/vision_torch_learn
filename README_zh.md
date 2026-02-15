# Pytorch 学习实战

## 概述
这个项目是我学习Pytorch深度学习的边缘端视觉模型开发，有3个子项目作为递进式学习的台阶：
- CIFAR-10 数据集的图像分类，掌握 PyTorch 基础流程
- 在树莓派 4 B 上部署 YOLOv 5-Lite 模型，实现 30 FPS 以上的实时目标检测
- 在树莓派 4 B 上实现模型量化与剪枝，达到 35+ FPS 的极致性能

## ✨ 进展
- 成功建立pytorch环境
- 成功拟合 cos（x）
![cos(x)show](./cos.png)

## 🚀 快速开始
如果你需要利用这些材料学习，可以通过一下方式进行。**强烈建议安装Astral uv来帮助环境配置**

```bash
# 克隆这个仓库
git clone https://github.com/octpus-01/vision_torch_learn.git

cd vision_torch_learn

# 用 uv 包管理工具重构环境
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# 测试cos（x）拟合
uv run cos_try.py

# 预期生成cos.png

```

## 📁 项目结构


## 🤝 如何贡献

欢迎提交 Issue 或 Pull Request！  
请确保代码格式一致，并通过所有测试。

1. Fork 本项目
2. 创建你的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

