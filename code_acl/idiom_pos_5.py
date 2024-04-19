import matplotlib.pyplot as plt
import numpy as np

# 提供的数据
categories = ['ADV', 'ADP', 'VERB', 'ADJ', 'NOUN', 'PRON', 'PROPN']
mem_counts_7b = np.array([31, 66, 52, 37, 463, 26, 8])
non_mem_counts_7b = np.array([2, 6, 7, 5, 98, 6, 3])
mem_ratios_7b = np.array([0.9393939393939394, 0.9166666666666666, 0.8813559322033898,
                       0.8809523809523809, 0.8253119429590018, 0.8125, 0.7272727272727273])

# categories_7b_chat = ['ADV', 'ADP', 'VERB', 'ADJ', 'NOUN', 'PRON', 'PROPN']
mem_counts_7b_chat = np.array([28, 59, 46, 34, 420, 22, 8])
non_mem_counts_7b_chat = np.array([4, 14, 9, 5, 104, 7, 3])
mem_ratios_7b_chat = np.array([0.875, 0.8082191780821918, 0.8363636363636363,
                       0.8717948717948718, 0.8015267175572519, 0.7586206896551724, 0.7272727272727273])

# categories_13b = ['ADV', 'ADP', 'VERB', 'ADJ', 'NOUN', 'PRON', 'PROPN']
mem_counts_13b = np.array([28, 67, 53, 36, 473, 28, 9])
non_mem_counts_13b = np.array([4, 5, 7, 6, 84, 4, 2])
mem_ratios_13b = np.array([0.875, 0.9305555555555556, 0.8833333333333333,
                       0.8571428571428571, 0.8491921005385996, 0.875, 0.8181818181818182])

# categories_13b_chat = ['ADV', 'ADP', 'VERB', 'ADJ', 'NOUN', 'PRON', 'PROPN']
mem_counts_13b_chat = np.array([27, 65, 40, 34, 397, 20, 7])
non_mem_counts_13b_chat = np.array([7, 8, 21, 8, 186, 13, 4])
mem_ratios_13b_chat = np.array([0.7941176470588235, 0.8904109589041096, 0.6557377049180327,
                       0.8095238095238095, 0.6809605488850772, 0.6060606060606061, 0.6363636363636364])

size = "13b-chat"

if size == "7b":
    mem_counts = mem_counts_7b
    non_mem_counts = non_mem_counts_7b
    mem_ratios = mem_ratios_7b
elif size == "13b":
    mem_counts = mem_counts_13b
    non_mem_counts = non_mem_counts_13b
    mem_ratios = mem_ratios_13b
elif size == "7b-chat":
    mem_counts = mem_counts_7b_chat
    non_mem_counts = non_mem_counts_7b_chat
    mem_ratios = mem_ratios_7b_chat
elif size == "13b-chat":
    mem_counts = mem_counts_13b_chat
    non_mem_counts = non_mem_counts_13b_chat
    mem_ratios = mem_ratios_13b_chat

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

plt.savefig('idiom_POS_{}.pdf'.format(size), dpi=300, bbox_inches='tight')
plt.show()
