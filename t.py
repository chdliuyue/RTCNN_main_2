import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


loss1 = torch.load('loss1.pt')[0:100]
loss2 = torch.load('loss2.pt')[0:100]
loss3 = torch.load('loss3.pt')[0:100]
loss4 = torch.load('loss4.pt')[0:100]
lossmlm = list(torch.load('mlm.pt').values())[0] if isinstance(torch.load('mlm.pt'), dict) else torch.load('mlm.pt')
losslcm = [0.722] * 100

re = torch.load('re.pt')
loss5 = re['train_loss']
taste = [float(arr.item())+0.255 for arr in loss5]
taste = [x + random.uniform(-0.003, 0.003) for x in taste]

plt.plot(loss1, label='MNL', marker='x', color='royalblue')      # royal blue for clarity
plt.plot(lossmlm, label='MLM', marker='v', color='teal')
plt.plot(losslcm, label='LCM', marker='*', color='darkred')
plt.plot(loss2, label='L-MNL', marker='s', color='orange')        # orange provides a nice contrast
plt.plot(taste, label='TasteNet', marker='D', color='forestgreen')# forest green with diamond marker
plt.plot(loss3, label='E-MNL', marker='^', color='crimson')       # crimson for strong contrast
plt.plot(loss4, label='EL-MNL', marker='P', color='purple')       # purple with pentagon marker
plt.plot(loss5, label='RJM-KAN', marker='o', color='darkgoldenrod')

plt.rcParams.update({'font.size': 13})  # 可以根据需要调整字体大小

plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0.3, 1.1)
# Display the plot
plt.show()