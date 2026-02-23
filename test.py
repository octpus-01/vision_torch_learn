
from torchvision.datasets import CIFAR10

try:
    dataset = CIFAR10(root='data/cifar10', train=True, download=False)
    print(f"✅ 成功加载 {len(dataset)} 张训练图像")
    print(f"第一张图类型: {type(dataset[0][0])}")  # 应该是 <class 'PIL.Image.Image'>
except Exception as e:
    print(f"❌ 加载失败: {e}")