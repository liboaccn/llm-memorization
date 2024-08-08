import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_width = 0.25

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

    plt.savefig('../figure/term_context_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {3: 30, 4: 20},
        'N': {3: 124, 4: 34, 5: 13, 6: 4},
    }
    gemma_7b = {
        'Y': {3: 58, 4: 39, 5: 2},
        'N': {3: 96, 4: 15, 5: 11, 6: 4},
    }
    llama2_7b = {
        'Y': {3: 22, 4: 16},
        'N': {3: 132, 4: 38, 5: 13, 6: 4},
    }
    llama2_13b = {
        'Y': {3: 46, 4: 17, 5: 3},
        'N': {3: 108, 4: 37, 5: 10, 6: 4},
    }
    llama3_8b = {
        'Y': {3: 50, 4: 34, 5: 4},
        'N': {3: 104, 4: 20, 5: 9, 6: 4},
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