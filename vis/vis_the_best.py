import os
import json
import matplotlib.pyplot as plt

def load_best_result_by_test(folder_path):
    best_result = None
    highest_pearson = -float('inf')
    best_file_path = ''

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('_results.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)

            current_pearson = data['results']['test']['pearson_corr_without_mean']
            
            if current_pearson > highest_pearson:
                highest_pearson = current_pearson
                best_result = data['results']
                best_file_path = os.path.basename(file_path)  # 保存文件名用于标签

    return best_result, best_file_path

def plot_all_metrics_combined(best_results_by_target, results_dir):
    metrics = ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    targets = ['y1', 'y2', 'y3']
    
    for idx, target in enumerate(targets):
        for path, results in best_results_by_target[target].items():
            values = [results[metric]['pearson_corr_without_mean'] for metric in metrics]
            axs[idx].plot(metrics, values, label=path.split('_')[-1])  # 标签使用文件名的最后一部分
            
        axs[idx].set_title(f'Target {target}')
        axs[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for ax in axs:
        ax.set_xlabel('Metrics')
    axs[0].set_ylabel('Pearson Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'combined_best_pearson_correlation.png'))
    plt.show()

results_dir = '../results'
best_results_by_target = {target: {} for target in ['y1', 'y2', 'y3']}

for subdir in os.listdir(results_dir):
    subdir_path = os.path.join(results_dir, subdir)
    if os.path.isdir(subdir_path):
        for target in best_results_by_target.keys():
            if subdir.endswith(target):
                best_result, best_file_path = load_best_result_by_test(subdir_path)
                if best_result:  # 确保存在结果
                    best_results_by_target[target][subdir] = best_result

plot_all_metrics_combined(best_results_by_target, results_dir)

