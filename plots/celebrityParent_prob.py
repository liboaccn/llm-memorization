import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义两组正态分布的参数：均值和方差
mean1, variance1 = 0.8899446092687403, 0.00411332014954599  # memorized, predict parent
mean2, variance2 = 0.782851577496096, 0.006473464771311012  # non-memorized, predict child

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
plt.fill_between(x, y1, color="#6C946B", alpha=0.5)
plt.plot(x, y1, color="#6C946B", label="Memorized, $\mu = 0.8899, \sigma^2=0.0041$")

plt.fill_between(x, y2, color="#B8CAD6", alpha=0.5)
plt.plot(x, y2, color="#B8CAD6", label="Non-memorized, $\mu=0.7828, \sigma^2=0.0064$")


# 添加标题和标签,图例
plt.xlabel('predicted probability', fontsize=14)
# plt.ylabel('Probability Density')
plt.xticks(fontsize=14)
plt.legend(fontsize=10)

# 隐藏整个y轴（包括轴线和标签）
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().spines['left'].set_visible(False)  # 隐藏y轴的竖线

# 隐藏上方和右侧的边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('CelebrityParent_prob.pdf', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()
