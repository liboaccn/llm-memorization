import json
import numpy as np


def compute_cosine_similarity(array_a, array_b):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_list = []

    for i in range(array_a.shape[0]):
        sim = cosine_similarity(array_a[i].reshape(1, -1), array_b[i].reshape(1, -1))[0]
        similarity_list.extend(sim)

    return similarity_list

    # return np.dot(array_a, array_b) / (np.linalg.norm(array_a) * np.linalg.norm(array_b))


def compute_euclidean_distance(array_a, array_b):
    return np.linalg.norm(array_a - array_b)


def compute_inclination(array_a, array_b):
    return np.arccos(np.dot(array_a, array_b) / (np.linalg.norm(array_a) * np.linalg.norm(array_b)))


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


def plot_slop_graph(context_child, generated_parent, context_parent):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cosine_sim_1 = compute_cosine_similarity(generated_parent, context_child)
    cosine_sim_2 = compute_cosine_similarity(context_parent, context_child)

    plt.figure(figsize=(8, 6))
    # slope graph
    plt.plot(cosine_sim_1, label='Generated Parent v.s. Context Child')
    plt.plot(cosine_sim_2, label='Context Parent v.s. Context Child')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('slope_graph.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_histogram():
    pass


def plot_box_plot():
    pass

def plot_overlapping_density():
    pass


if __name__ == "__main__":
    r_file = "CelebrityParent_predict_child_hidden_v2.json"
    # step 2: similarity: (generated_parent_name, context_parent_name) in (mem vs. non-mem); Histogram; overlapping density
    # step 3: similarity: (generated_parent_name, context_child_name) v.s. similarity: (context_parent_name, context_child_name); slop graph; box plot

    context_child, generated_parent, context_parent = a(r_file)
    plot_slop_graph(context_child, generated_parent, context_parent)