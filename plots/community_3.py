# 全部的代码如下，遵循了您的设置要求

import matplotlib.pyplot as plt
import numpy as np

# 设置数据
integration_methods = ['$\mathbf{h}_u$', '$\mathbf{h}_n$', 'avg.', 'concat', 'x-attn']
tencent_auc = [0.9202, 0.9262, 0.9230, 0.9271, 0.9301]
tencent_ap = [0.9303, 0.9352, 0.9333, 0.9380, 0.9443]

x = np.arange(len(integration_methods))  # the label locations
bar_width = 0.36  # the width of the bars

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制AUC柱子
auc_bars = ax.bar(x - bar_width/2, tencent_auc, bar_width, label='AUC', color='white', edgecolor='#89A36D', hatch='xx')
# 绘制AP柱子
ap_bars = ax.bar(x + bar_width/2, tencent_ap, bar_width, label='AP', color='white', edgecolor='#F8CBAD', hatch='xx')

# 设置y轴的限制
plt.ylim(0.920, 0.946)

# 设置x轴和y轴标签与图例的字体大小
ax.set_ylabel('Scores', fontsize=16)
plt.title('Tencent', fontsize=16)
ax.legend(fontsize=16, loc='upper left')

# 设置刻度标签的字体大小
ax.set_xticks(x)
ax.set_xticklabels(integration_methods, fontsize=16)
plt.yticks([])

# 为每个柱子添加数值标签
for bar in auc_bars + ap_bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

fig.tight_layout()
plt.savefig('integration_tencent.pdf', dpi=300, bbox_inches='tight')
plt.show()






##################### sina
# 设置数据
integration_methods = ['$\mathbf{h}_u$', '$\mathbf{h}_n$', 'avg.', 'concat', 'x-attn']
sina_auc = [0.7665, 0.7796, 0.7721, 0.7801, 0.7943]
sina_ap = [0.8295, 0.8467, 0.8341, 0.8512, 0.8768]

x = np.arange(len(integration_methods))  # the label locations
# bar_width = 0.36  # the width of the bars

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制AUC柱子
auc_bars = ax.bar(x - bar_width/2, sina_auc, bar_width, label='AUC', color='white', edgecolor='#89A36D', hatch='xx')

# 绘制AP柱子
ap_bars = ax.bar(x + bar_width/2, sina_ap, bar_width, label='AP', color='white', edgecolor='#F8CBAD', hatch='xx')

# 设置y轴的限制
plt.ylim(0.762, 0.885)

# 设置x轴和y轴标签与图例的字体大小
# ax.set_xlabel('Integration Method', fontsize=14)
ax.set_ylabel('Scores', fontsize=16)
plt.title('Sina', fontsize=16)
ax.legend(fontsize=16)

# 设置刻度标签的字体大小
ax.set_xticks(x)
ax.set_xticklabels(integration_methods, fontsize=16)
# ax.set_yticklabels(ax.get_yticks(), fontsize=14)
plt.yticks([])

# 为每个柱子添加数值标签
for bar in auc_bars + ap_bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

fig.tight_layout()
plt.savefig('integration_sina.pdf', dpi=300, bbox_inches='tight')
plt.show()

