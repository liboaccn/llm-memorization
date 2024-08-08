import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_width = 0.35

    ax1.bar(categories, y_values, width=bar_width, label='Memorized', color='moccasin')
    ax1.bar(categories, n_values, width=bar_width, bottom=y_values, label='Non-memorized', color='sandybrown')
    ax1.set_xlabel('Context length', fontsize=20)
    ax1.set_xticks(categories)
    ax1.set_xticklabels(categories, fontsize=18)
    ax1.set_ylabel('Number of samples', fontsize=20)

    ax2 = ax1.twinx()
    ax2.plot(categories, ratios, label='Ratio of non-memorized', color='salmon', marker='o', linewidth=2)
    ax2.set_ylabel('Ratio of non-memorized', fontsize=20)
    ax2.set_ylim(0, 1)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=16)

    plt.savefig('../figure/proper_context_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {4: 63, 5: 35, 6: 21, 7: 6, 8: 3},
        'N': {4: 81, 5: 32, 6: 20, 7: 12, 8: 6},
    }
    gemma_7b = {
        'Y': {4: 93, 5: 51, 6: 30, 7: 17, 8: 5},
        'N': {4: 51, 5: 16, 6: 11, 7: 1, 8: 4},
    }
    llama2_7b = {
        'Y': {4: 29, 5: 19, 6: 8, 7: 3, 8: 1},
        'N': {4: 115, 5: 48, 6: 33, 7: 15, 8: 8},
    }
    llama2_13b = {
        'Y': {4: 83, 5: 49, 6: 30, 7: 11, 8: 4},
        'N': {4: 61, 5: 18, 6: 11, 7: 7, 8: 5},
    }
    llama3_8b = {
        'Y': {4: 83, 5: 49, 6: 34, 7: 18, 8: 9},
        'N': {4: 61, 5: 18, 6: 7},
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