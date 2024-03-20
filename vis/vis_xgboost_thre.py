import json
import matplotlib.pyplot as plt
import os
import re

def load_results(folder_path):
    results = {}
    for folder_name in os.listdir(folder_path):
        match = re.search(r'xgboost_LR_(y\d+|у\d+)_(\d+\.\d+|\d+)', folder_name)
        if match:
            target, threshold = match.groups()
            threshold = float(threshold)

            if target not in results:
                results[target] = {
                    'thresholds': [],
                    'train': [],
                    'val': [],
                    'test': [],
                    'train_pos': [],
                    'val_pos': [],
                    'test_pos': []
                }

            file_path = os.path.join(folder_path, folder_name, "results.json")
            with open(file_path, 'r') as file:
                data = json.load(file)

            results[target]['thresholds'].append(threshold)
            for key in ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']:
                results[target][key].append(data['results'][key]['pearson_corr_without_mean'])

    return results

folder_path = '../results'  # 请替换为实际的文件夹路径
results = load_results(folder_path)

# 结果类型和对应的颜色
result_types = ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# 对每个目标绘图
for target in results.keys():
    plt.figure(figsize=(10, 6))
    for i, result_type in enumerate(result_types):
        plt.scatter(results[target]['thresholds'], results[target][result_type], color=colors[i], label=result_type)
    
    plt.title(f'Pearson Correlation vs Threshold for {target}')
    plt.xlabel('Threshold')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    plt.show()
