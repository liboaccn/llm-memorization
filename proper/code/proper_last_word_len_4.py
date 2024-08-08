import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    # categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    categories = ("<2", '<6', '<11', '>11')
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    # Calculating the ratio for 'N' as a percentage of the total (N + Y)
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    # 创建条形图和折线图的组合图形
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 创建条形图
    bar_width = 0.3

    # Plotting the 'Y' values
    ax1.bar(categories, y_values, width=bar_width, label='Memorized', color='#1f77b4')
    ax1.bar(categories, n_values, width=bar_width, bottom=y_values, label='Non-memorized', color='#ff7f0e')
    ax1.set_xlabel('Predicted length', fontsize=20)
    ax1.set_xticks(categories)
    ax1.set_xticklabels(categories, fontsize=17)
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

    plt.savefig('../figure/proper_predicted_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {2: 20, 5: 33, 8: 55, 11: 22},
        'N': {2: 40, 5: 61, 8: 32, 11: 18},
    }
    gemma_7b = {
        'Y': {2: 51, 3: 41, 4: 8, 5: 18, 6: 17, 7: 14, 8: 23, 9: 5, 10: 8, 11: 6, 12: 3, 13: 4},
        'N': {2: 9, 3: 16, 4: 2, 5: 9, 6: 10, 7: 7, 8: 16, 9: 2, 10: 5, 11: 2, 13: 4, 14: 1},
    }
    llama2_7b = {
        'Y': {2: 2, 3: 8, 4: 2, 5: 13, 6: 10, 7: 9, 8: 8, 9: 2, 10: 3, 11: 1, 12: 1, 13: 1},
        'N': {2: 58, 3: 49, 4: 8, 5: 14, 6: 17, 7: 12, 8: 31, 9: 5, 10: 10, 11: 7, 12: 2, 13: 7, 14: 1},
    }
    llama2_13b = {
        'Y': {"<2": 39, "<6": 80, "<11": 54, ">11": 6},
        'N': {"<2": 21, "<6": 41, "<11": 34, ">11": 6},
    }
    llama3_8b = {
        'Y': {2: 44, 3: 45, 4: 7, 5: 18, 6: 19, 7: 14, 8: 25, 9: 4, 10: 8, 11: 4, 12: 2, 13: 4},
        'N': {2: 16, 3: 12, 4: 3, 5: 9, 6: 8, 7: 7, 8: 14, 9: 3, 10: 5, 11: 4, 12: 1, 13: 4, 14: 1},
    }

    variables = {
        'mistral_7b': mistral_7b,
        'gemma_7b': gemma_7b,
        'llama2_7b': llama2_7b,
        'llama2_13b': llama2_13b,
        'llama3_8b': llama3_8b
    }

    for name, data in variables.items():
        plotting(data, model_name_or_path=name)