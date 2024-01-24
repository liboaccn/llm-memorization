import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 假设的128维向量
v1 = np.random.rand(128)
v2 = np.random.rand(128)

# 将两个向量合并为一个2x128的矩阵
data = np.vstack([v1, v2])

# 应用PCA降至二维
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

# 绘制结果
plt.figure()
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=['red', 'blue'])
plt.text(transformed_data[0, 0], transformed_data[0, 1], 'v1', fontsize=12)
plt.text(transformed_data[1, 0], transformed_data[1, 1], 'v2', fontsize=12)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of 128-dimensional vectors')
plt.grid(True)
plt.show()
