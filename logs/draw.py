import re
import matplotlib.pyplot as plt

file_path = 'logs/log.txt'
with open(file_path, 'r') as file:
    data = file.read()

loss_pattern = r'"Loss": (\d+\.\d+)'
loss_values = re.findall(loss_pattern, data)

# 将 Loss 值转换为浮点数
loss_values = [float(loss) for loss in loss_values]

# 每隔 10 个取一个 Loss 值
loss_values_sampled = loss_values[::10]  # 切片操作，步长为 10

# 生成对应的 epoch 列表
epochs_sampled = list(range(1, len(loss_values) + 1))[::10]  # 同样每隔 10 个取一个

# 绘制 Loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs_sampled, loss_values_sampled, label='Loss (Sampled every 10 epochs)', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs (Sampled every 10 epochs)')
plt.grid(True)
plt.legend()
plt.show()