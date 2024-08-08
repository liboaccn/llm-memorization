import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = set(data['Y'].keys()).union(data['N'].keys())
    categories = ('<5', '<8', '<12', '>12')
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

    plt.savefig('../figure/term_predicted_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {"<5": 23, "<8": 19, "<12": 8},
        'N': {"<5": 63, "<8": 47, "<12": 44, ">12": 21},
    }
    # gemma_7b = {
    #     'Y': {5: 47, 8: 31, 12: 13, 14: 8},
    #     'N': {5: 39, 8: 35, 12: 39, 14: 13},
    # }
    gemma_7b = {
        'Y': {"<5": 47, "<8": 31, "<12": 13, ">12": 8},
        'N': {"<5": 39, "<8": 35, "<12": 39, ">12": 13},
    }
    # llama2_7b = {
    #     'Y': {5: 21, 8: 11, 12: 6},
    #     'N': {5: 65, 8: 55, 12: 46, 14: 21},
    # }
    llama2_7b = {
        'Y': {"<5": 21, "<8": 11, "<12": 6},
        'N': {"<5": 65, "<8": 55, "<12": 46, ">12": 21},
    }
    # llama2_13b = {
    #     'Y': {5: 29, 8: 23, 12: 14},
    #     'N': {5: 57, 8: 43, 12: 38, 14: 21},
    # }
    llama2_13b = {
        'Y': {"<5": 29, "<8": 23, "<12": 14},
        'N': {"<5": 57, "<8": 43, "<12": 38, ">12": 21},
    }
    # llama3_8b = {
    #     'Y': {5: 43, 8: 27, 12: 13, 14: 5},
    #     'N': {5: 43, 8: 39, 12: 39, 14: 16},
    # }
    llama3_8b = {
        'Y': {"<5": 43, "<8": 27, "<12": 13, ">12": 5},
        'N': {"<5": 43, "<8": 39, "<12": 39, ">12": 16},
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