import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义两组正态分布的参数：均值和方差
mean1, variance1 = -0.0021384479814851045, 1.60325496665004  # memorized
mean2, variance2 = 0.004812135891848817, 1.7529506512942592  # non-memorized

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
plt.fill_between(x, y1, color="lightblue", alpha=0.5)
plt.plot(x, y1, color="blue", label="Memorized, $\mu=-0.0021, \sigma^2=1.6032$")

plt.fill_between(x, y2, color="lightgreen", alpha=0.5)
plt.plot(x, y2, color="green", label="Non-memorized, $\mu=0.0048, \sigma^2=1.7529$")


# 添加标题和标签,图例
plt.xlabel('predicted hidden state', fontsize=14)
# plt.ylabel('Probability Density')
plt.xticks(fontsize=14)
plt.legend(fontsize=10)

# 隐藏整个y轴（包括轴线和标签）
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().spines['left'].set_visible(False)  # 隐藏y轴的竖线

# 隐藏上方和右侧的边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('idiom_hidden.pdf', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()
