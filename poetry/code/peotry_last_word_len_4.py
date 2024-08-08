import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    # Calculating the ratio for 'N' as a percentage of the total (N + Y)
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    # 创建条形图和折线图的组合图形
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 创建条形图
    bar_width = 0.1

    # Plotting the 'Y' values
    ax1.bar(categories, y_values, width=bar_width, label='Memorized', color='#1f77b4')
    ax1.bar(categories, n_values, width=bar_width, bottom=y_values, label='Non-memorized', color='#ff7f0e')
    ax1.set_xlabel('Predicted length', fontsize=20)
    ax1.set_xticks(categories)
    ax1.set_xticklabels(categories, fontsize=18)
    ax1.set_ylabel('Number of samples', fontsize=20)

    # Creating a secondary y-axis for the ratios
    ax2 = ax1.twinx()
    ax2.plot(categories, ratios, label='Ratio of non-memorized', color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Ratio of non-memorized', fontsize=20)
    ax2.set_ylim(0, 1.01)

    # Moving the legends together
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=16)

    plt.savefig('../figure/peotry_predicted_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    qwen1_5_7b = {
        'Y':{1: 896, 2: 621},
        'N': {1: 545, 2: 820},
    }
    qwen1_5_14b = {
        'Y': {1: 1012, 2: 712},
        'N': {1: 429, 2: 729},
    }
    qwen1_5_32b_chat = {
        'Y': {1: 984, 2:698},
        'N': {1: 457, 2:743},
    }

    variables = {
        'qwen1_5_7b': qwen1_5_7b,
        'qwen1_5_14b': qwen1_5_14b,
        'qwen1_5_32b_chat': qwen1_5_32b_chat,
    }

    for name, data in variables.items():
        plotting(data, model_name_or_path=name)