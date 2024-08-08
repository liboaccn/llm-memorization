import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.bar(categories, y_values, label='Memorized', color='moccasin')
    ax1.bar(categories, n_values, bottom=y_values, label='Non-memorized', color='sandybrown')
    ax1.set_xlabel('context length', fontsize=18)
    ax1.set_xticks(categories)
    ax1.set_xticklabels(categories, fontsize=18)
    ax1.set_ylabel('Number of samples', fontsize=18)

    ax2 = ax1.twinx()
    ax2.plot(categories, ratios, label='Ratio of non-memorized', color='salmon', marker='o', linewidth=2)
    ax2.set_ylabel('Ratio of non-memorized', fontsize=18)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=14)

    plt.savefig('../figure/popQA_context_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {4: 3, 5: 19, 6: 72, 7: 61, 8: 19, 9: 8, 10: 3},
        'N': {4: 13, 5: 31, 6: 137, 7: 134, 8: 35, 9: 11, 10: 3},
    }
    gemma_7b = {
        'Y': {4: 2, 5: 8, 6: 79, 7: 77, 8: 21, 9: 8},
        'N': {4: 14, 5: 42, 6: 130, 7: 118, 8: 33, 9: 11},
    }
    llama2_7b = {
        'Y': {4: 1, 5: 15, 6: 69, 7: 56, 8: 18, 9: 8},
        'N': {4: 15, 5: 35, 6: 140, 7: 139, 8: 36, 9: 11},
    }
    llama2_13b = {
        'Y': {4: 1, 5: 21, 6: 90, 7: 62, 8: 16, 9: 8},
        'N': {4: 15, 5: 29, 6: 119, 7: 133, 8: 38, 9: 11},
    }
    llama3_8b = {
        'Y': {4: 2, 5: 21, 6: 76, 7: 72, 8: 19, 9: 10},
        'N': {4: 14, 5: 29, 6: 133, 7: 123, 8: 35, 9: 9},
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
