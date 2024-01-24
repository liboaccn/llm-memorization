import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Example data: Replace these with your actual data
# Let's say each group has 10 vectors of 128 dimensions
group_a = np.random.rand(100, 128)
group_b = np.random.rand(700, 128)

# Combine the groups
combined_data = np.vstack((group_a, group_b))

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, n_iter=1000, random_state=0)
tsne_results = tsne.fit_transform(combined_data)

# Split the results back into two groups
tsne_a = tsne_results[:len(group_a)]
tsne_b = tsne_results[len(group_a):]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', label='Group A')
plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color='blue', label='Group B')
plt.legend()
plt.title('t-SNE visualization of two groups')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
