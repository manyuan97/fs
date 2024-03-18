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

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith('k_') and file_name.endswith('_results.json'):
            k = int(file_name.split('_')[1])
            ks.append(k)

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
            for key in results:
                results[key].append(data['results'][key]['pearson_corr_without_mean'])
    return ks, results


# Paths to the folders
folder_paths = [
    '../results/f_regression_cross_LR_y1',
    '../results/f_regression_cross_LR_y2',
    '../results/f_regression_cross_LR_y3'
]

# Colors for the plots
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

for i, folder_path in enumerate(folder_paths):
    ks, results = load_results(folder_path)

    for j, key in enumerate(['train', 'val', 'test']):
        axs[0, i].plot(ks, results[key], label=key.capitalize(), color=colors[j])

    axs[0, i].set_xlabel('Number of Selected Features (k)')
    axs[0, i].set_ylabel('Pearson Correlation (without mean)')
    axs[0, i].set_title(f'{os.path.basename(folder_path)}\nStandard')
    axs[0, i].legend()
    axs[0, i].grid(True)

    # Plotting the second group ('train_pos', 'val_pos', 'test_pos')
    for j, key in enumerate(['train_pos', 'val_pos', 'test_pos']):
        axs[1, i].plot(ks, results[key], label=key.capitalize(), color=colors[j + 3],marker='x')

    axs[1, i].set_xlabel('Number of Selected Features (k)')
    axs[1, i].set_ylabel('Pearson Correlation (without mean)')
    axs[1, i].set_title(f'{os.path.basename(folder_path)}\nPositive')
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.savefig('./pearson_correlation_plots_grouped.png')
plt.show()
