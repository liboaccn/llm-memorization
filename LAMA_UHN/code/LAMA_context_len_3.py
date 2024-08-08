import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_width = 0.3

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

    plt.savefig('../figure/LAMA_context_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mistral_7b = {
        'Y': {'< 10': 12, '< 17': 59, '< 25': 67, '> 25': 61},
        'N': {'< 10': 67, '< 17': 122, '< 25': 85, '> 25': 54},
    }
    gemma_7b = {
        'Y': {'< 10': 22, '< 17': 76, '< 25': 91, '> 25': 80},
        'N': {'< 10': 57, '< 17': 105, '< 25': 61, '> 25': 35},
    }
    llama2_7b = {
        'Y': {'< 10': 3, '< 17': 29, '< 25': 49, '> 25': 37},
        'N': {'< 10': 76, '< 17': 152, '< 25': 103, '> 25': 78},
    }
    llama2_13b = {
        'Y': {'< 10': 14, '< 17': 51, '< 25': 52, '> 25': 50},
        'N': {'< 10': 65, '< 17': 130, '< 25': 100, '> 25': 65},
    }
    llama3_8b = {
        'Y': {'< 10': 25, '< 17': 76, '< 25': 82, '> 25': 65},
        'N': {'< 10': 54, '< 17': 105, '< 25': 70, '> 25': 50},
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
