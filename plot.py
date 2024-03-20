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

    for folder_name in sorted(os.listdir(folder_path), key=lambda x: int(''.join([i for i in x if i.isdigit()]))):
        k = int(''.join([i for i in folder_name if i.isdigit()]))  # 提取数字
        ks.append(k)

        file_path = os.path.join(folder_path, folder_name, "results.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        for key in results:
            results[key].append(data['results'][key]['pearson_corr_without_mean'])
    return ks, results

# 颜色设置
colors = ['b', 'g', 'r']
pos_colors = ['c', 'm', 'y']

# 绘图
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列

ks, results = load_results('../results')  # 假设所有相关文件夹都在此路径下

# 标准结果
for i, key in enumerate(['train', 'val', 'test']):
    axs[0, i].plot(ks, results[key], label=key.capitalize(), color=colors[i], marker='o')
    axs[0, i].set_title(f'{key.capitalize()} Pearson Correlation')
    axs[0, i].set_xlabel('Data Ratio (%)')
    axs[0, i].set_ylabel('Pearson Correlation (without mean)')
    axs[0, i].grid(True)
    axs[0, i].legend()

# 正向结果
for i, key in enumerate(['train_pos', 'val_pos', 'test_pos']):
    axs[1, i].plot(ks, results[key], label=key.capitalize(), color=pos_colors[i], marker='x')
    axs[1, i].set_title(f'{key.capitalize()} Pearson Correlation')
    axs[1, i].set_xlabel('Data Ratio (%)')
    axs[1, i].set_ylabel('Pearson Correlation (without mean)')
    axs[1, i].grid(True)
    axs[1, i].legend()

plt.tight_layout()
plt.savefig('./pearson_correlation_vs_data_ratio.png')
plt.show()
