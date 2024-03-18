import json
import matplotlib.pyplot as plt

# 模拟输入路径
json_file_paths = [
    'results/xgboost_LR_y1_cross/results.json',
    'results/xgboost_LR_y2_cross/results.json',
    'results/xgboost_LR_y3_cross/results.json'
]

# 设置图表标题和文件名
plot_titles = []
train_mse_values = []
val_mse_values = []
n_features_values = []

for json_file_path in json_file_paths:
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    n_features = [item['n_features'] for item in data['results']['cross_result']]
    train_mse = [item['train_metrics']['pearson_corr_without_mean'] for item in data['results']['cross_result']]
    val_mse = [item['val_metrics']['pearson_corr_without_mean'] for item in data['results']['cross_result']]

    n_features_values.append(n_features)
    train_mse_values.append(train_mse)
    val_mse_values.append(val_mse)

    file_title = json_file_path.split('/')[-2]
    plot_titles.append(file_title)

plt.figure(figsize=(15, 7))

colors = ['r', 'g', 'b']
line_styles = ['-', '--']

for i in range(len(json_file_paths)):
    plt.plot(n_features_values[i], train_mse_values[i], line_styles[0], color=colors[i], marker='o', label=f'{plot_titles[i]} Train pearson_corr_without_mean')
    plt.plot(n_features_values[i], val_mse_values[i], line_styles[1], color=colors[i], marker='x', label=f'{plot_titles[i]} Val pearson_corr_without_mean')

plt.title('MSE vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('MSE')
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend(loc='upper right')
combined_title = '_'.join(plot_titles)
plt.savefig(f'{combined_title}.png')
plt.close()

print(f"Combined plot saved as {combined_title}.png")
