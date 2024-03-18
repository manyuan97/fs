import json
import matplotlib.pyplot as plt
import os


def load_results(folder_path):
    results = {
        'train': [],
        'val': [],
        'test': [],
        'train_pos': [],
        'val_pos': [],
        'test_pos': []
    }
    ks = []
    selected_features = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith('thre_') and file_name.endswith('.json'):
            k = float(file_name.split('_')[1].split('.json')[0])
            ks.append(k)

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)

            for key in results:
                results[key].append(data['results'][key]['pearson_corr_without_mean'])

            selected_features.append(data['n_features'])

    return ks, results, selected_features


# Paths to the folders
folder_paths = [
    '../results/xgboost_xgboost_cross_y1',
    '../results/xgboost_xgboost_cross_y2',
    '../results/xgboost_xgboost_cross_y3'
]

# Colors for the plots
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

for i, folder_path in enumerate(folder_paths):
    ks, results, selected_features = load_results(folder_path)

    for j, key in enumerate(['train', 'val', 'test']):
        ax = axs[0, i]
        ax.plot(ks, results[key], label=key.capitalize(), color=colors[j],marker='x')

        ax.set_xlabel('Threshold (k)')
        ax.set_ylabel('Pearson Correlation (without mean)')
        ax.set_title(f'{os.path.basename(folder_path)}\nStandard')
        ax.grid(True)

    sec_ax = ax.twinx()
    sec_ax.plot(ks, selected_features, 'k--', label='Selected Features')
    sec_ax.set_ylabel('Number of Selected Features')

    lines, labels = ax.get_legend_handles_labels()
    sec_lines, sec_labels = sec_ax.get_legend_handles_labels()
    ax.legend(lines + sec_lines, labels + sec_labels, loc='upper left')

    for j, key in enumerate(['train_pos', 'val_pos', 'test_pos']):
        ax = axs[1, i]
        ax.plot(ks, results[key], label=key.capitalize(), color=colors[j + 3],marker='x')

        ax.set_xlabel('Threshold (k)')
        ax.set_ylabel('Pearson Correlation (without mean)')
        ax.set_title(f'{os.path.basename(folder_path)}\nPositive')
        ax.grid(True)

    sec_ax = ax.twinx()
    sec_ax.plot(ks, selected_features, 'k--', label='Selected Features')
    sec_ax.set_ylabel('Number of Selected Features')

    lines, labels = ax.get_legend_handles_labels()
    sec_lines, sec_labels = sec_ax.get_legend_handles_labels()
    ax.legend(lines + sec_lines, labels + sec_labels, loc='upper left')

plt.tight_layout()
plt.savefig('./xgboost_correlation_plots_grouped_with_features.png')
plt.show()