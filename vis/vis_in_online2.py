import json
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = '../results'

alpha_values = []
pearson_corr_values = {
    'train': [], 'val': [], 'test': [],
    'train_pos': [], 'val_pos': [], 'test_pos': []
}

targets = ['y1', 'y2', 'y3']
for target in targets:
    all_alphas_file_path = os.path.join(base_dir, f'lasso_cross_{target}', 'all_alphas.json')

    with open(all_alphas_file_path, 'r', encoding='utf-8') as json_file:
        all_alphas_data = json.load(json_file)

        for key in pearson_corr_values.keys():
            pearson_corr_values[key].append([])

        for alpha_key, metrics in all_alphas_data.items():
            alpha = float(alpha_key.split('_')[1])
            if alpha not in alpha_values:
                alpha_values.append(alpha)

            for dataset in pearson_corr_values.keys():
                pearson_corr_values[dataset][-1].append(metrics['results'][dataset]['pearson_corr'])

alpha_values = np.array(alpha_values)
sorted_indices = np.argsort(alpha_values)
sorted_alpha_values = alpha_values[sorted_indices]

fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

datasets = ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']
for i, dataset in enumerate(datasets):
    row = i // 3
    col = i % 3

    for target_index, target in enumerate(targets):
        sorted_pearson_corr_values = np.array(pearson_corr_values[dataset][target_index])[sorted_indices]

        axs[row, col].plot(sorted_alpha_values, sorted_pearson_corr_values, '-o', label=target)
        axs[row, col].set_title(f'Alpha vs Pearson Correlation ({dataset})')
        axs[row, col].set_xlabel('Alpha')
        axs[row, col].set_xscale('log')
        axs[row, col].grid(True)
        if col == 0:
            axs[row, col].set_ylabel('Pearson Correlation')

    axs[row, col].legend()

plt.savefig(os.path.join(base_dir, 'alpha_vs_pearson_corr_grid.png'))
plt.close()

print("Grid plot saved as alpha_vs_pearson_corr_grid.png in the results directory.")