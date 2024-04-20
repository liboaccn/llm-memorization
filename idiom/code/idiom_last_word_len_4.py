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
    bar_width = 0.35

    # Plotting the 'Y' values
    ax1.bar(categories, y_values, label='Memorized', color='#1f77b4')
    ax1.bar(categories, n_values, bottom=y_values, label='Non-memorized', color='#ff7f0e')
    ax1.set_xlabel('Length of the last word', fontsize=14)
    ax1.set_xticks(categories)
    ax1.set_xticklabels(categories, fontsize=14)
    ax1.set_ylabel('Number of samples', fontsize=14)

    # Creating a secondary y-axis for the ratios
    ax2 = ax1.twinx()
    ax2.plot(categories, ratios, label='Ratio of non-memorized', color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Ratio of non-memorized', fontsize=14)

    # Moving the legends together
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)

    plt.savefig('idiom_last_word_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    llama2_7b = {
        'Y': {2: 62, 3: 118, 4: 205, 5: 168, 6: 67, 7: 42, 8: 28, 9: 3, 10: 2, 11: 1},
        'N': {2: 8, 3: 16, 4: 39, 5: 23, 6: 18, 7: 13, 8: 5, 9: 5},
    }

    llama2_7b_chat = {
        'Y': {2: 54, 3: 110, 4: 191, 5: 149, 6: 61, 7: 35, 8: 22, 9: 5, 10: 1, 11: 1},
        'N': {2: 12, 3: 22, 4: 42, 5: 29, 6: 18, 7: 13, 8: 5, 9: 5, 11: 1},
    }

    llama2_13b = {
        'Y': {2: 65, 3: 118, 4: 210, 5: 167, 6: 72, 7: 42, 8: 28, 9: 4, 10: 2},
        'N': {2: 6, 3: 16, 4: 33, 5: 23, 6: 16, 7: 12, 8: 3, 9: 4},
    }

    llama2_13b_chat = {
        'Y': {2: 61, 3: 104, 4: 184, 5: 141, 6: 60, 7: 26, 8: 22, 9: 3, 10: 1, 11: 1},
        'N': {2: 11, 3: 35, 4: 62, 5: 53, 6: 29, 7: 36, 8: 12, 9: 8, 10: 1, 11: 1, 12: 1},
    }

    gemma_7b = {

    }

    mistral_7b = {}

    qwen15_7b = {}
    qwen15_14b = {}

    for data in [llama2_7b, llama2_13b,
                 # llama2_7b_chat, llama2_13b_chat,
                 # gemma_7b, mistral_7b, qwen15_7b, qwen15_14b
                 ]:
        plotting(data, model_name_or_path="{data}")