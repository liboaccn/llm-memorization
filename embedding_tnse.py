import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
# Example data: Replace these with your actual data
# Let's say each group has 10 vectors of 128 dimensions

idiom_y = []
idiom_n = []

explanation_y = []
explanation_n = []
# Read in the data from a file
with open('edata.jsonl', 'r') as f:
# with open('idiomem_demo.jsonl', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        idiom_v = data['idiom_v']
        explanation_v = data['explanation_v']
        match = data['match']
        if match == 'Y':
            idiom_y.append(idiom_v)
            explanation_y.append(explanation_v)
        else:
            idiom_n.append(idiom_v)
            explanation_n.append(explanation_v)

group_idiom_y = np.array(idiom_y)
group_explanation_y = np.array(explanation_y)
group_idiom_n = np.array(idiom_n)
group_explanation_n = np.array(explanation_n)

# group_a = np.random.rand(5, 12)
# group_b = np.random.rand(5, 12)
# print(group_a)
# exit(0)
# Combine the groups
combined_data = np.vstack((group_idiom_y, group_idiom_n))

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, n_iter=1000, random_state=0)
tsne_results = tsne.fit_transform(combined_data)

# Split the results back into two groups
tsne_a = tsne_results[:len(group_idiom_y)]
tsne_b = tsne_results[len(group_idiom_n):]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', label='Group Y')
plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color='blue', label='Group N')
plt.legend()
plt.title('t-SNE visualization of two groups')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()