import matplotlib.pyplot as plt

# Given data
data = {
    'Y': {2: 62, 3: 118, 4: 205, 5: 168, 6: 67, 7: 42, 8: 28, 9: 3},
    'N': {2: 8, 3: 16, 4: 39, 5: 23, 6: 18, 7: 13, 8: 5, 9: 5},
}

# Prepare data for plotting
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

plt.savefig('idiom_last_word_len.pdf', dpi=300, bbox_inches='tight')
plt.show()
