import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 示例数据
group_a = np.random.rand(5, 32)
group_b = np.random.rand(10, 32)

# 标准化向量
group_a_normalized = normalize(group_a)
group_b_normalized = normalize(group_b)

# 计算余弦相似度
cos_sim = cosine_similarity(np.vstack((group_a_normalized, group_b_normalized)))

# 使用PCA降维到2维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(np.vstack((group_a_normalized, group_b_normalized)))

# 绘制矢量图
plt.figure(figsize=(8, 6))
for point in reduced_data[:5]:
    plt.quiver(0, 0, point[0], point[1], angles='xy', scale_units='xy', scale=1, color='red')
for point in reduced_data[5:]:
    plt.quiver(0, 0, point[0], point[1], angles='xy', scale_units='xy', scale=1, color='blue')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('Vector Visualization of Two Groups')
plt.grid(True)
plt.show()
