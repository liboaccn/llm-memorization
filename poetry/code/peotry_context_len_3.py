import matplotlib.pyplot as plt


def plotting(data, model_name_or_path):
    # categories = sorted(set(data['Y'].keys()).union(data['N'].keys()))
    categories = ('4', '6')
    y_values = [data['Y'].get(category, 0) for category in categories]
    n_values = [data['N'].get(category, 0) for category in categories]
    ratios = [(n / (y + n)) if (y + n) > 0 else 0 for y, n in zip(y_values, n_values)]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_width = 0.1

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

    plt.savefig('../figure/peotry_context_len_{}.pdf'.format(model_name_or_path), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    qwen1_5_7b = {
        'Y': {4: 341, 6: 543},
        'N': {4: 282, 6: 259},
    }
    qwen1_5_14b = {
        'Y': {4: 402, 6: 597},
        'N': {4: 221, 6: 205},
    }
    qwen1_5_32b_chat = {
        'Y': {"4": 397, "6": 577},
        'N': {"4": 226, "6": 225},
    }

    variables = {
        'qwen1_5_7b': qwen1_5_7b,
        'qwen1_5_14b': qwen1_5_14b,
        'qwen1_5_32b_chat': qwen1_5_32b_chat,
    }

    for name, data in variables.items():
        plotting(data, model_name_or_path=name)
