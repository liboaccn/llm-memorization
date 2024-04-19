# import matplotlib.pyplot as plt
# import numpy as np
# age_groups = ['AUC', 'AP']
#
# r = [0.9008, 0.9037]
# e = [0.9101, 0.9153]
# t = [0.9208, 0.9264]
# r_e = [0.9240, 0.9274]
# t_r = [0.9254, 0.9286]
# t_e = [0.9231, 0.9299]
# h = [0.9202, 0.9303]
#
# fig, ax = plt.subplots(figsize=(8, 6))
# bar_width = 0.1
# index = np.arange(len(age_groups))
#
# bars_r = ax.bar(index - 3.0*bar_width, r, bar_width, label='$\mathbf{r}$', color='#8D9FD1')
# bars_e = ax.bar(index - 2.0*bar_width, e, bar_width, label='$\mathbf{e}$', color='#D18CB8')
# bars_t = ax.bar(index - 1.0*bar_width, t, bar_width, label='$\mathbf{t}$', color='#FFBB78')
# bars_r_e = ax.bar(index + 0.0*bar_width, r_e, bar_width, label='$\mathbf{r}+\mathbf{e}$', color='#89A36D')
# bars_t_r = ax.bar(index + 1.0*bar_width, t_r, bar_width, label='$\mathbf{t}+\mathbf{r}$', color='#F8CBAD')
# bars_t_e = ax.bar(index + 2.0*bar_width, t_e, bar_width, label='$\mathbf{t}+\mathbf{e}$', color='#AEC7E8')
# bars_h = ax.bar(index + 3.0*bar_width, h, bar_width, label='$\mathbf{h}_u$', color='#8F267E')
#
# # ax.set_xlabel('Age (years)', fontsize=14)
# ax.set_ylabel('Scores', fontsize=14)
# ax.set_title('Tencent', fontsize=16)
# ax.set_xticks(index)
# ax.set_xticklabels(age_groups, fontsize=14)
# plt.ylim(0.9002, 0.9305)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14, loc='best')
# plt.savefig('nodefeature_tencent.pdf', dpi=300, bbox_inches='tight')
# plt.show()




########################
import matplotlib.pyplot as plt
import numpy as np

age_groups = ['AUC', 'AP']

r = [0.7398, 0.8676]
e = [0.7459, 0.8787]
t = [0.7586, 0.8812]
r_e = [0.7590, 0.8830]
t_r = [0.7606, 0.8845]
t_e = [0.7647, 0.8871]
h = [0.7665, 0.8895]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.1
index = np.arange(len(age_groups))

bars_r = ax.bar(index - 3.0*bar_width, r, bar_width, label='$\mathbf{r}$', color='#8D9FD1')
bars_e = ax.bar(index - 2.0*bar_width, e, bar_width, label='$\mathbf{e}$', color='#D18CB8')
bars_t = ax.bar(index - 1.0*bar_width, t, bar_width, label='$\mathbf{t}$', color='#FFBB78')
bars_r_e = ax.bar(index + 0.0*bar_width, r_e, bar_width, label='$\mathbf{r}+\mathbf{e}$', color='#89A36D')
bars_t_r = ax.bar(index + 1.0*bar_width, t_r, bar_width, label='$\mathbf{t}+\mathbf{r}$', color='#F8CBAD')
bars_t_e = ax.bar(index + 2.0*bar_width, t_e, bar_width, label='$\mathbf{t}+\mathbf{e}$', color='#AEC7E8')
bars_h = ax.bar(index + 3.0*bar_width, h, bar_width, label='$\mathbf{h}_u$', color='#8F267E')

ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Sina', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(age_groups, fontsize=14)
plt.ylim(0.7380, 0.89)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, loc='best')
plt.savefig('nodefeature_sina.pdf', dpi=300, bbox_inches='tight')
plt.show()