import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Type': ['acc', 'prob', 'acc', 'prob', 'acc', 'prob', 'acc', 'prob', 'acc', 'prob'],
    'Model': ['LLaMA 2 7B', 'LLaMA 2 7B', 'Mistral 7B', 'Mistral 7B', 'LLaMA 2 13B', 'LLaMA 2 13B',
              'LLaMA 3 8B', 'LLaMA 3 8B', 'Gemma 7B', 'Gemma 7B'],
    # 'IDIOM': [0.817, 0.381, 0.711, 0.566, 0.831, 0.409, 0.561, 0.326, 0.803, 0.376],
    'ProperNoun': [0.214, 0.394, 0.463, 0.534, 0.637, 0.607, 0.69, 0.683, 0.705, 0.721],
    'Terminology': [0.169, 0.542, 0.222, 0.606, 0.293, 0.612, 0.391, 0.648, 0.44, 0.602],
    'PopQA': [0.31, 0.676, 0.337, 0.672, 0.368, 0.648, 0.37, 0.668, 0.361, 0.64],
    'LAMA-UHN': [0.224, 0.409, 0.378, 0.57, 0.317, 0.54, 0.471, 0.62, 0.51, 0.604]
}

# Create DataFrame
df = pd.DataFrame(data)

# Pivot data for easier plotting
pivot_df = df.melt(id_vars=['Model', 'Type'], var_name='Dataset', value_name='Score')

datasets = pivot_df['Dataset'].unique()

fig, ax1 = plt.subplots(figsize=(12, 8))

# Line styles and colors - improved for better visual appeal
colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

# Adjusting line styles and thickness
line_styles = ['-', '--', '-.', ':']
line_widths = [4, 4]  # Thicker lines for better visibility
marker = ['*', 'o', 'D', 'v', 'P']

# Creating plots for each dataset with improved aesthetics
for i, dataset in enumerate(datasets):
    subset_acc = pivot_df[(pivot_df['Type'] == 'acc') & (pivot_df['Dataset'] == dataset)]
    subset_prob = pivot_df[(pivot_df['Type'] == 'prob') & (pivot_df['Dataset'] == dataset)]
    ax1.plot(subset_acc['Model'], subset_acc['Score'], label=f'{dataset} Acc', color=colors[i % len(colors)], linestyle=line_styles[0], linewidth=line_widths[0], marker=marker[0], markersize=16)
    ax1.plot(subset_prob['Model'], subset_prob['Score'], label=f'{dataset} Prob', color=colors[i % len(colors)], linestyle=line_styles[1], linewidth=line_widths[1], marker=marker[1], markersize=18)

# Configuring axes for aesthetics
# ax1.set_xlabel('Model', fontsize=14)
ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=20)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

# Right axis for probabilities
ax2 = ax1.twinx()
ax2.set_ylabel('Probability', color='tab:red', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=20)

# Improving legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(datasets), fontsize=16, title_fontsize='13')

# Adding a grid for better readability
ax1.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.savefig('../figure/model_datasets_acc_prob.pdf', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
