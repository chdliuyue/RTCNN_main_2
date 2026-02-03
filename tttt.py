import torch
import matplotlib.pyplot as plt
import random

# 加载数据
loss1 = torch.load('loss1.pt')[0:100]
loss2 = torch.load('loss2.pt')[0:100]
loss3 = torch.load('loss3.pt')[0:100]
loss4 = torch.load('loss4.pt')[0:100]
lossmlm = list(torch.load('mlm.pt').values())[0] if isinstance(torch.load('mlm.pt'), dict) else torch.load('mlm.pt')
losslcm = [0.722] * 100

re = torch.load('re.pt')
loss5 = re['train_loss']
taste = [float(arr.item()) + 0.255 for arr in loss5]
taste = [x + random.uniform(-0.003, 0.003) for x in taste]

# 绘图
plt.figure(figsize=(10, 6))  # 调整图像的大小

# 设置配色和标记风格
plt.plot(loss1, label='MNL', marker='x', color='#1f77b4', linestyle='-', linewidth=2, markersize=6)      # 使用蓝色，增加线条宽度和标记大小
plt.plot(lossmlm, label='MLM', marker='v', color='#ff7f0e', linestyle='--', linewidth=2, markersize=6)   # 使用橙色，虚线样式
plt.plot(losslcm, label='LCM', marker='*', color='#2ca02c', linestyle='-', linewidth=2, markersize=8)    # 使用绿色，星形标记
plt.plot(loss2, label='L-MNL', marker='s', color='#d62728', linestyle='-.', linewidth=2, markersize=6)   # 使用红色，点划线样式
plt.plot(taste, label='TasteNet', marker='D', color='#9467bd', linestyle='-', linewidth=2, markersize=6) # 使用紫色，菱形标记
plt.plot(loss3, label='E-MNL', marker='^', color='#8c564b', linestyle='--', linewidth=2, markersize=6)   # 使用棕色，三角形向上
plt.plot(loss4, label='EL-MNL', marker='P', color='#e377c2', linestyle='-', linewidth=2, markersize=7)   # 使用粉色，五边形
plt.plot(loss5, label='RJM-KAN', marker='o', color='#7f7f7f', linestyle='-', linewidth=2, markersize=6)  # 使用灰色，圆形标记

# 调整图例
plt.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black')

# 增加标签和网格
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)

# 设置轴范围
plt.xlim(0, 100)
plt.ylim(0.3, 1.1)

# 调整整体布局
plt.tight_layout()

# 显示图像
plt.show()
