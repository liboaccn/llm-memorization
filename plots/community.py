import matplotlib.pyplot as plt

# 给定的数据
# communities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# log_likelihood_per_1000 = [-19.07, -18.77, -18.02, -17.95, -17.35, -16.90, -16.09, -15.59, -15.20, -15.11, -15.33, -15.58, -16.02, -17.39, -19.90, -20.43, -20.53, -20.70, -21.90, -22.00]
#
# # 根据新的要求绘制折线图并保存
# plt.figure(figsize=(8, 6))
# plt.plot(communities, log_likelihood_per_1000, marker='^', linestyle='-', color='#8F267E', markersize=10)
# plt.xlabel('Number of Communities', fontsize=14)
# plt.ylabel('Log-likelihood/1000', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.ylim(-23, -7)
# plt.grid(True)
# plt.savefig('number_c_tencent.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# ---------------
# import matplotlib.pyplot as plt
#
# # 给定的数据
# communities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# log_likelihood_per_1000 = [-15.97, -15.87, -15.82, -15.70, -14.65, -14.20, -13.89, -13.59, -12.20, -11.51,
#                            -10.73, -9.28, -8.72, -8.00, -7.80, -7.93, -8.03, -8.70, -9.90, -12.50]
#
# # 根据新的要求绘制折线图并保存
# plt.figure(figsize=(8, 6))
# plt.plot(communities, log_likelihood_per_1000, marker='^', linestyle='-', color='#8F267E', markersize=10)
# plt.xlabel('Number of Communities', fontsize=14)
# plt.ylabel('Log-likelihood/1000', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.ylim(-23, -7)
# plt.grid(True)
# plt.savefig('number_c_sina.pdf', dpi=300, bbox_inches='tight')
# plt.show()
#---------------
# 由于代码执行环境重置，需要重新导入matplotlib.pyplot和设置数据及图形属性
import matplotlib.pyplot as plt

# 设置数据
communities_tencent = [50, 100, 150, 200]
auc = [0.9091, 0.9301, 0.8930, 0.8011]
ap = [0.9256, 0.9443, 0.9004, 0.8278]

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(communities_tencent, auc, marker='^', linestyle='--', linewidth=3, color='#8D9FD1', markersize=12, label='AUC')
plt.plot(communities_tencent, ap, marker='*', linestyle='--', linewidth=3, color='#D18CB8', markersize=12, label='AP')
plt.xlabel('Number of Communities', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.xticks(communities_tencent, communities_tencent)  # 设置x轴只显示四个点的tick
plt.grid(True)
plt.ylim(0.69, 0.95)
plt.legend()
plt.savefig('score_tencent.pdf')
plt.show()


import matplotlib.pyplot as plt

# 设置数据
communities_sina = [50, 100, 150, 200]
auc = [0.7009, 0.7789, 0.7943, 0.7757]
ap = [0.7992, 0.8912, 0.9068, 0.8897]

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(communities_tencent, auc, marker='^', linestyle='--', linewidth=3, color='#8D9FD1', markersize=12, label='AUC')
plt.plot(communities_tencent, ap, marker='*', linestyle='--', linewidth=3, color='#D18CB8', markersize=12, label='AP')
plt.xlabel('Number of Communities', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.xticks(communities_sina, communities_sina)  # 设置x轴只显示四个点的tick
plt.grid(True)
plt.ylim(0.69, 0.947)
plt.legend()
plt.savefig('score_sina.pdf')
plt.show()