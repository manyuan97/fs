import json
import matplotlib.pyplot as plt
import os


def load_results(folder_path, alphas):
    results = {
        'train': [],
        'val': [],
        'test': [],
        'train_pos': [],
        'val_pos': [],
        'test_pos': []
    }
    selected_features = []

    for alpha in sorted(alphas):
        file_name = f'alpha_{alpha}.json'
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)

        for key in results:
            results[key].append(data['results'][key]['pearson_corr_without_mean'])

        selected_features.append(data['n_features'])

    return results, selected_features


# Paths to the folders
folder_paths = [
    '../results/lasso_cross_y1',
    '../results/lasso_cross_y2',
    '../results/lasso_cross_y3'
]

alphas = [0.00001, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.1]

colors = ['b', 'g', 'r', 'c', 'm', 'y']

fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

for i, folder_path in enumerate(folder_paths):
    results, selected_features = load_results(folder_path, alphas)

    for j, key in enumerate(['train', 'val', 'test']):
        axs[0, i].plot(alphas, results[key], label=key.capitalize(), color=colors[j])

    axs[0, i].set_xscale('log')
    axs[0, i].set_xlabel('Alpha Value')
    axs[0, i].set_ylabel('Pearson Correlation (without mean)')
    axs[0, i].set_title(f'{os.path.basename(folder_path)}\nStandard')
    axs[0, i].legend()
    axs[0, i].grid(True)

    sec_ax = axs[0, i].twinx()
    sec_ax.plot(alphas, selected_features, 'k--', label='Selected Features')
    sec_ax.set_ylabel('Number of Selected Features')
    sec_ax.legend(loc='lower left')

    for j, key in enumerate(['train_pos', 'val_pos', 'test_pos']):
        axs[1, i].plot(alphas, results[key], label=key.capitalize(), color=colors[j + 3])

    axs[1, i].set_xscale('log')
    axs[1, i].set_xlabel('Alpha Value')
    axs[1, i].set_ylabel('Pearson Correlation (without mean)')
    axs[1, i].set_title(f'{os.path.basename(folder_path)}\nPositive')
    axs[1, i].legend()
    axs[1, i].grid(True)

    sec_ax = axs[1, i].twinx()
    sec_ax.plot(alphas, selected_features, 'k--', label='Selected Features')
    sec_ax.set_ylabel('Number of Selected Features')
    sec_ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('./lasso_correlation_alpha_plots_grouped.png')
plt.show()

