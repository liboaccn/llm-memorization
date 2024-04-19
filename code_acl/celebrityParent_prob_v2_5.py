import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义两组正态分布的参数：均值和方差
# model = '7b'
# model = '13b'
# model = '7b-chat'
# model = '13b-chat'

# model = '7b-random'
# model = '13b-random'
# model = '7b-chat-random'
model = '13b-chat-random'
if model == '7b':
    mean1, variance1 = 0.8568, 0.0044  # memorized, predict parent
    mean2, variance2 = 0.6502, 0.0114  # non-memorized, predict child
elif model == '13b':
    mean1, variance1 = 0.8851, 0.0039
    mean2, variance2 = 0.6707, 0.0113
elif model == '7b-chat':
    mean1, variance1 = 0.9205, 0.0032
    mean2, variance2 = 0.7717, 0.0101
elif model == '13b-chat':
    mean1, variance1 = 0.9406, 0.0021
    mean2, variance2 = 0.7985, 0.0063
elif model == '7b-random':
    mean1, variance1 = 0.8550, 0.0045
    mean2, variance2 = 0.8594, 0.0048
elif model == '13b-random':
    mean1, variance1 = 0.8854, 0.0040
    mean2, variance2 = 0.8845, 0.0039
elif model == '7b-chat-random':
    mean1, variance1 = 0.9217, 0.0029
    mean2, variance2 = 0.9221, 0.0030
elif model == '13b-chat-random':
    mean1, variance1 = 0.9380, 0.0022
    mean2, variance2 = 0.9384, 0.0023


# 计算标准差
std_dev1 = np.sqrt(variance1)
std_dev2 = np.sqrt(variance2)

# 生成x的值，考虑到两个分布的范围
x = np.linspace(min(mean1 - 3*std_dev1, mean2 - 3*std_dev2), max(mean1 + 3*std_dev1, mean2 + 3*std_dev2), 1000)

# 计算两个正态分布的PDF值
y1 = norm.pdf(x, mean1, std_dev1)
y2 = norm.pdf(x, mean2, std_dev2)

# 设置图形大小为8x6英寸
plt.figure(figsize=(8, 6))

# 绘制两个正态分布
plt.fill_between(x, y1, color="#fab0c4", alpha=0.5)
plt.plot(x, y1, color="#fab0c4", label="Memorized, $\mu={}, \sigma^2={}$".format(mean1, variance1))

plt.fill_between(x, y2, color="#B8CAD6", alpha=0.5)
plt.plot(x, y2, color="#B8CAD6", label="Non-memorized, $\mu={}, \sigma^2={}$".format(mean2, variance2))


# 添加标题和标签,图例
plt.xlabel('predicted probability', fontsize=14)
plt.xlim(0.3, 1.1)
plt.xticks(fontsize=14)
plt.legend(fontsize=10)

# 隐藏整个y轴（包括轴线和标签）
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().spines['left'].set_visible(False)  # 隐藏y轴的竖线

# 隐藏上方和右侧的边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('CelebrityParent_prob_v2_{}.pdf'.format(model), dpi=300, bbox_inches='tight')
# 显示图形
plt.show()
