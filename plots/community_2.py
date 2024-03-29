from matplotlib import pyplot as plt

# 设置数据
models = ['w/o $\mathbf{h}_n$', 'w/o $\mathbf{h}_u$', 'Ours']
tencent_auc = [0.920, 0.926, 0.930]
tencent_ap = [0.930, 0.935, 0.944]

sina_auc = [0.766, 0.779, 0.794]
sina_ap = [0.829, 0.846, 0.876]

x = range(len(models))  # the label locations

# 绘制Tencent的柱状图
plt.figure(figsize=(6, 4))
# 设置柱状图的宽度
bar_width = 0.3
# AUC
auc_bars = plt.barh(x, tencent_auc, height=bar_width, label='AUC', color='white', edgecolor='#8D9FD1', hatch='///')
# AP，在AUC的基础上向右移动一定的宽度
ap_bars = plt.barh([i + bar_width for i in x], tencent_ap, height=bar_width, label='AP', color='white', hatch='///', edgecolor='#D18CB8')
# 为每个柱子标识数值
for bar in auc_bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}',
             va='center', ha='left')

for bar in ap_bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}',
             va='center', ha='left')

# 设置y轴的刻度标签
plt.yticks([i + bar_width/2 for i in x], models, fontsize=16)
plt.xticks([])
plt.xlim(0.92, 0.95)
plt.legend(fontsize=14)
plt.title('Tencent', fontsize=16)
plt.tight_layout()
plt.savefig('ablation_tencent.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 绘制Sina的柱状图
plt.figure(figsize=(6, 4))
bar_width = 0.3
# AUC
auc_bars = plt.barh(x, sina_auc, height=bar_width, label='AUC', color='white', edgecolor='#8D9FD1', hatch='///')
# AP，在AUC的基础上向右移动一定的宽度
ap_bars = plt.barh([i + bar_width for i in x], sina_ap, height=bar_width, label='AP', color='white', hatch='///', edgecolor='#D18CB8')
# 为每个柱子标识数值
for bar in auc_bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}',
             va='center', ha='left')

for bar in ap_bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}',
             va='center', ha='left')
# 设置y轴的刻度标签
plt.yticks([i + bar_width/2 for i in x], models, fontsize=16)
plt.xticks([])
plt.xlim(0.76, 0.91)
plt.legend(loc='lower right', fontsize=14)
plt.title('Sina', fontsize=16)
plt.tight_layout()
plt.savefig('ablation_sina.pdf', dpi=300, bbox_inches='tight')
plt.show()
