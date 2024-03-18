import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Base directory where the folders for each target variable are located
base_dir = './results'

# Initialize dictionaries to store the data for plotting
alpha_values = []
pearson_corr_values = {
    'train': [], 'val': [], 'test': [],
    'train_pos': [], 'val_pos': [], 'test_pos': []
}
num_features_values = {
    'y1': [], 'y2': [], 'y3': []
}

# Process each target variable by reading the all_alphas.json file
targets = ['y1', 'y2', 'y3']  # Target variables
for target in targets:
    # Path to the all_alphas.json file for the current target variable
    all_alphas_file_path = os.path.join(base_dir, f'lasso_cross_{target}', 'all_alphas.json')

    # Read the all_alphas.json file
    with open(all_alphas_file_path, 'r', encoding='utf-8') as json_file:
        all_alphas_data = json.load(json_file)

        # Initialize lists for current target
        num_features_values[target] = []
        for key in pearson_corr_values.keys():
            pearson_corr_values[key].append([])

        # Iterate through each alpha's results
        for alpha_key, metrics in all_alphas_data.items():
            alpha = float(alpha_key.split('_')[1])
            if alpha not in alpha_values:
                alpha_values.append(alpha)

            # Append the pearson_corr for the current alpha
            selected_features = metrics['selected_features']
            num_features_values[target].append(sum(metrics['selected_features']))
            for dataset in pearson_corr_values.keys():
                pearson_corr_values[dataset][-1].append(metrics['results'][dataset]['pearson_corr'])

# Sort alpha values for plotting
alpha_values = np.array(alpha_values)
sorted_indices = np.argsort(alpha_values)
sorted_alpha_values = alpha_values[sorted_indices]

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

# Define colors for each target
colors = {'y1': 'blue', 'y2': 'green', 'y3': 'red'}

# Plot the pearson_corr and number of selected features for each dataset
datasets = ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']
for i, dataset in enumerate(datasets):
    row = i // 3  # Determine row index
    col = i % 3   # Determine column index

    for target_index, target in enumerate(targets):
        # Sort the pearson_corr based on sorted alpha values
        sorted_pearson_corr_values = np.array(pearson_corr_values[dataset][target_index])[sorted_indices]
        sorted_num_features_values = np.array(num_features_values[target])[sorted_indices]

        ax2 = axs[row, col].twinx()  # Create a second y-axis for the number of selected features
        ax2.plot(sorted_alpha_values, sorted_num_features_values, 'o--', color=colors[target], label=f'{target} Num Features')
        ax2.set_ylabel('Num Features', color=colors[target])
        ax2.tick_params(axis='y', labelcolor=colors[target])

        axs[row, col].plot(sorted_alpha_values, sorted_pearson_corr_values, 'o-', color=colors[target], label=f'{dataset} {target} Pearson Corr')
        axs[row, col].set_title(f'Alpha vs Metrics ({dataset})')
        axs[row, col].set_xlabel('Alpha')
        axs[row, col].set_xscale('log')
        axs[row, col].grid(True)
        if col == 0 and row == 0:
            axs[row, col].set_ylabel('Pearson Correlation')

    axs[row, col].legend(loc='upper left')
    ax2.legend(loc='upper right')

# Save the entire figure
plt.savefig(os.path.join(base_dir, 'alpha_vs_metrics_and_features_grid.png'))
plt.close()

print("Grid plot saved as alpha_vs_metrics_and_features_grid.png in the results directory.")

