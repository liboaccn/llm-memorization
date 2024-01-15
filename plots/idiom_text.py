import matplotlib.pyplot as plt
import numpy as np

# 提供的数据
categories = ['ADV', 'ADP', 'VERB', 'ADJ', 'NOUN', 'PRON', 'PROPN']
mem_counts = np.array([31, 66, 52, 37, 463, 26, 8])
non_mem_counts = np.array([2, 6, 7, 5, 98, 6, 3])
mem_ratios = np.array([0.9393939393939394, 0.9166666666666666, 0.8813559322033898,
                       0.8809523809523809, 0.8253119429590018, 0.8125, 0.7272727272727273])

# 计算总数用于条形图
total_counts = mem_counts + non_mem_counts

# 创建条形图和折线图的组合图形
fig, ax1 = plt.subplots(figsize=(8, 6))

# 创建条形图
bar_width = 0.35
index = np.arange(len(categories))

bar1 = ax1.bar(index, mem_counts, bar_width, label='Memorized', color='skyblue')
bar2 = ax1.bar(index + bar_width, non_mem_counts, bar_width, label='Non-memorized', color='coral')

# 设置图形的X轴和标题
ax1.set_xlabel('Part-of-speech', fontsize=14)
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(categories, fontsize=14)

# 创建折线图
ax2 = ax1.twinx()
line1, = ax2.plot(index + bar_width / 2, mem_ratios, color='darkgreen', label='Ratio of Memorized', linewidth=2, marker='o')


# 设置图形的Y轴和标题
ax1.set_ylabel('Number of samples', fontsize=14)
ax1.set_ylim(0, 500)
ax1.set_yticks(np.arange(0, 501, 100))
ax1.set_yticklabels(np.arange(0, 501, 100), fontsize=14)

# Y-axis settings for line chart (denser tick marks)
ax2.set_ylabel('Ratio of Memorized', fontsize=14)
ax2.set_ylim(0.7, 1)
ax2.set_yticks(np.linspace(0.7, 1, 6))  # denser ticks
ax2.set_yticklabels(np.round(np.linspace(0.7, 1, 6), 2), fontsize=12)

# 设置图例
# Combined legend for both bar and line charts
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)

plt.savefig('idiom_POS.pdf', dpi=300, bbox_inches='tight')
plt.show()
