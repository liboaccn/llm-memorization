import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def compute_cosine_similarity(array_a, array_b):
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_list = []
    for i in range(array_a.shape[0]):
        sim = cosine_similarity(array_a[i].reshape(1, -1), array_b[i].reshape(1, -1))[0]
        similarity_list.extend(sim)
    return similarity_list


def compute_euclidean_distance(array_a, array_b):
    distances = np.sqrt(np.sum((array_a - array_b) ** 2, axis=1))
    return distances


def compute_inclination(array_a, array_b):
    dot_product = np.sum(array_a * array_b, axis=1)
    norm_a = np.linalg.norm(array_a, axis=1)
    norm_b = np.linalg.norm(array_b, axis=1)
    cos_theta = dot_product / (norm_a * norm_b)
    angles = np.arccos(cos_theta)  # This will be in radians
    return angles


def a(r_file):
    context_child_list = []
    generated_parent_list = []
    context_parent_list = []

    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            context_child = data['context_child_hidden']
            generated_parent = data['gen_parent_hidden']
            context_parent = data['context_parent_hidden']

            context_child_list.append(context_child)
            generated_parent_list.append(generated_parent)
            context_parent_list.append(context_parent)

    context_child_list = np.asarray(context_child_list)
    generated_parent_list = np.array(generated_parent_list)
    context_parent_list = np.array(context_parent_list)

    return context_child_list, generated_parent_list, context_parent_list


def plot_box_plot(context_child, generated_parent, context_parent):
    cosine_sim_1 = compute_cosine_similarity(context_child, generated_parent)  # A
    cosine_sim_2 = compute_cosine_similarity(context_child, context_parent)  # B
    cosine_sim_3 = compute_cosine_similarity(context_parent, generated_parent)  # C

    print(sum(cosine_sim_1) / len(cosine_sim_1))
    print(sum(cosine_sim_2) / len(cosine_sim_2))
    print(sum(cosine_sim_3) / len(cosine_sim_3))

    plt.figure(figsize=(8, 6))
    plt.boxplot([cosine_sim_1, cosine_sim_2, cosine_sim_3], labels=['$A$', '$B$', '$C$'],
                showmeans=False, meanline=True, meanprops={'color': 'r', 'linestyle': '-', 'linewidth': 3},
                notch=False, patch_artist=False,
                boxprops=dict(linewidth=3),
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3, color='k')
                )
    plt.ylabel('Cosine Similarity', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14,)
    plt.savefig('box_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------
def plot_slope_graph_euclidean(context_child, generated_parent, context_parent):
    eu_1 = compute_euclidean_distance(context_child, generated_parent)
    eu_2 = compute_euclidean_distance(context_child, context_parent)
    eu_3 = compute_euclidean_distance(context_parent, generated_parent)


    print(sum(eu_1) / len(eu_1))
    print(sum(eu_2) / len(eu_2))
    print(sum(eu_3) / len(eu_3))

    s_sim_1, s_sim_2, s_sim_3 = [], [], []
    prob = []
    for i in range(len(eu_1)):
        prob.append(eu_1[i] - eu_2[i])
    min_prob = min(prob)
    if min_prob < 0:
        prob = [p + abs(min_prob) for p in prob]
    prob = np.array(prob) / sum(prob)

    # Sampling 2 items based on the normalized probabilities
    sampled_indices = np.random.choice(len(prob), size=80, replace=False, p=prob)
    for i in sampled_indices:
        s_sim_1.append(eu_1[i])
        s_sim_2.append(eu_2[i])
        s_sim_3.append(eu_3[i])
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    x = ['$A$', '$B$', '$C$']
    y = [s_sim_1, s_sim_2, s_sim_3]

    # Plotting the slope graph
    plt.plot(x, y, marker='o', linestyle='-', color='#501B8A', linewidth=1.5, markersize=4)

    # plt.xlabel
    plt.ylabel('Euclidean Distance', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Showing the plot
    plt.savefig('slope_graph_euclidean.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_overlapping_density_euclidean(context_child, generated_parent, context_parent):
    eu_1 = compute_euclidean_distance(generated_parent, context_child)
    eu_2 = compute_euclidean_distance(context_parent, context_child)

    print(sum(eu_1) / len(eu_1))
    print(sum(eu_2) / len(eu_2))

    plt.figure(figsize=(8, 6))

    # Plotting the density plots
    sns.kdeplot(eu_1, label='(Generated Parent, Context Child)', shade=True, color="r", alpha=0.5)
    sns.kdeplot(eu_2, label='(Context Parent, Context Child)', shade=True, color="b", alpha=0.5)

    # Adding labels and title
    plt.xlabel('Euclidean Distance', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=10)

    plt.savefig('overlapping_density_euclidean.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------
def plot_violin_inclination(context_child, generated_parent, context_parent):
    in_1 = compute_inclination(context_child, generated_parent)
    in_2 = compute_inclination(context_child, context_parent)
    in_3 = compute_inclination(context_parent, generated_parent)

    print(sum(in_1) / len(in_1))
    print(sum(in_2) / len(in_2))
    print(sum(in_3) / len(in_3))

    plt.figure(figsize=(8, 6))
    plt.violinplot([in_1, in_2, in_3], showmeans=True)
    plt.xticks([1, 2, 3], ['$A$', '$B$', '$C$'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Inclination', fontsize=16)
    plt.savefig('violin_density_inclination.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    r_file = "CelebrityParent_predict_child_hidden_v2.json"
    context_child, generated_parent, context_parent = a(r_file)

    # cosine similarity
    plot_box_plot(context_child, generated_parent, context_parent)

    # euclidean distance
    plot_slope_graph_euclidean(context_child, generated_parent, context_parent)
    # plot_overlapping_density_euclidean(context_child, generated_parent, context_parent)

    # inclination
    plot_violin_inclination(context_child, generated_parent, context_parent)
