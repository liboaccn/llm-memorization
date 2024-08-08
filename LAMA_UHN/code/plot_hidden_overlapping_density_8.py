import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def calculate_mean_h(r_file, match='Y'):
    with open(r_file, 'r') as f:
        all_mean = []
        all_var = []
        count = 0
        for i, line in enumerate(f):
            count = i+1
            data = json.loads(line)
            if data['match'] == match:
                hidden = np.array(data['mean_hidden'])
                mean = hidden.mean()
                var = hidden.var()

                all_mean.append(mean)
                all_var.append(var)
    return all_var


def eu_distance(r_file, match='Y'):
    with open(r_file, 'r') as f:
        all_norm = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                hidden = np.array(data['mean_hidden'])
                norm = np.linalg.norm(hidden)
                all_norm.append(norm)
    return all_norm


def get_hidden(r_file, match='Y'):
    with open(r_file, 'r', encoding='utf-8') as f:
        all_hidden = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                hidden = data['mean_hidden']
                all_hidden.append(hidden)
    all_hidden = np.array(all_hidden)

    return all_hidden


def plot_overlapping_density_euclidean(model, r_file):
    # eu_1 = calculate_mean_h(r_file, match='Y')
    # eu_2 = calculate_mean_h(r_file, match='N')

    eu_1 = eu_distance(r_file, match='Y')
    eu_2 = eu_distance(r_file, match='N')

    print(sum(eu_1) / len(eu_1))
    print(sum(eu_2) / len(eu_2))

    plt.figure(figsize=(8, 6))

    # Plotting the density plots
    sns.kdeplot(eu_1, label='memorized', shade=True, color="r", alpha=0.5)
    sns.kdeplot(eu_2, label='non-memorized', shade=True, color="b", alpha=0.5)

    # Adding labels and title
    plt.xlabel('hidden xxx', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=10)

    plt.savefig('../figure/noun_overlapping_density_hidden_{}.pdf'.format(model), dpi=300, bbox_inches='tight')
    plt.show()


def plot_pca(model, r_file):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 将数据合并在一起用于PCA
    group1 = get_hidden(r_file, match='Y')
    group2 = get_hidden(r_file, match='N')
    data = np.vstack((group1, group2))
    labels = np.array([0] * group1.shape[0] + [1] * group2.shape[0])  # 用于区分两组

    # PCA降维到2D
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[labels == 0, 0], data_pca[labels == 0, 1], label='memorized', alpha=0.5, s=60)
    plt.scatter(data_pca[labels == 1, 0], data_pca[labels == 1, 1], label='non-memorized', alpha=0.5, s=60)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)

    plt.savefig('../figure/LAMA_pca_{}.pdf'.format(model), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    from load_LLMs import MODELS

    prompt_num = 10
    last_n = 1
    for model_name_or_path in MODELS:
        r_file = '../data/LAMA_UHN_out_{}_shot_{}.jsonl'.format(prompt_num, model_name_or_path.split('/')[-1])
        print('=========== {} ==========='.format(r_file), '\n')
        # plot_overlapping_density_euclidean(model_name_or_path.split('/')[-1], r_file)
        plot_pca(model_name_or_path.split('/')[-1], r_file)



