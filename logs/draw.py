import re
import matplotlib.pyplot as plt

file_path = 'logs/pro_log1-14.txt'
with open(file_path, 'r') as file:
    data = file.read()

loss_pattern = r'"Loss": (\d+\.\d+)'
loss_values = re.findall(loss_pattern, data)
loss_values = [float(loss) for loss in loss_values]
loss_values_sampled = loss_values[::10]  # 切片操作，步长为 10
epochs_sampled = list(range(1, len(loss_values) + 1))[::10]  # 同样每隔 10 个取一个

plt.figure(figsize=(10, 6))
plt.plot(epochs_sampled, loss_values_sampled, label='Loss (Sampled every 10 epochs)', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs (Sampled every 10 epochs)')
plt.grid(True)
plt.legend()
plt.show()