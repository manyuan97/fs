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

    for folder_name in sorted(os.listdir(folder_path), key=lambda x: float(''.join([i for i in x if i.isdigit() or i == '.']))):
        k = float(''.join([i for i in folder_name if i.isdigit() or i == '.']))  # 提取数字，支持浮点数
        ks.append(k)

        file_path = os.path.join(folder_path, folder_name, "results.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        for key in results:
            results[key].append(data['results'][key]['pearson_corr_without_mean'])
    return ks, results

# 载入结果
folder_path = '../results'  # 更新为您的实际路径
ks, results = load_results(folder_path)

# 绘图设置
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y']
labels = ['train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos']

# 绘制所有结果类型
for i, key in enumerate(labels):
    plt.plot(ks, results[key], label=key.capitalize(), color=colors[i], marker='o', linestyle='-')

plt.title('Pearson Correlation vs Data Ratio')
plt.xlabel('Data Ratio (%)')
plt.ylabel('Pearson Correlation (without mean)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./all_pearson_correlation_vs_data_ratio.png')
plt.show()
