import numpy as np
import pickle
import os
import onnxruntime as ort

def load_cifar10_test_data(data_dir):
    """
    从原始二进制文件加载 CIFAR-10 测试集
    返回: images (N, 32, 32, 3), labels (N,)
    """
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')
    
    # 原始数据是 (N, 3072)，按 channel-first 存储
    data = test_dict[b'data']          # shape: (10000, 3072)
    labels = test_dict[b'labels']      # list of 10000 ints
    
    # 转换为 (N, 3, 32, 32) -> (N, 32, 32, 3)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images.astype(np.float32), np.array(labels)

def normalize_image(image):
    """
    应用与训练时相同的归一化
    CIFAR-10 常用: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    return (image / 255.0 - mean) / std

def main():
    # 1. 加载测试数据
    print("正在加载 CIFAR-10 测试集...")
    images, labels = load_cifar10_test_data('cifar-10-batches-py')
    print(f"加载完成: {images.shape[0]} 张图像")
    
    # 2. 预处理：归一化 + 转为 channel-first (N, 3, 32, 32)
    print("正在预处理图像...")
    processed_images = np.stack([
        normalize_image(img).transpose(2, 0, 1)  # (32,32,3) -> (3,32,32)
        for img in images
    ], axis=0)  # shape: (10000, 3, 32, 32)
    
    # 3. 加载 ONNX 模型
    print("正在加载 ONNX 模型...")
    session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 4. 批量推理（避免内存溢出）
    print("开始推理...")
    batch_size = 16
    correct = 0
    total = 0
    
    for i in range(0, len(processed_images), batch_size):
        batch_images = processed_images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # 推理
        outputs = session.run([output_name], {input_name: batch_images})
        predictions = np.argmax(outputs[0], axis=1)  #type: ignore
        
        # 统计正确数
        correct += np.sum(predictions == batch_labels)
        total += len(batch_labels)
        
        if i % 1000 == 0:
            print(f"已处理 {total}/10000 张图像...")
    
    # 5. 计算准确率
    accuracy = correct / total
    print(f"\n✅ 最终准确率: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()