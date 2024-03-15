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
mse_values = []
n_features_values = []

# 遍历每个JSON文件路径
for json_file_path in json_file_paths:
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 提取cross_result中的n_features和mse
    n_features = [item['n_features'] for item in data['results']['cross_result']]
    mse = [item['mse'] for item in data['results']['cross_result']]

    # 保存每个文件的结果
    n_features_values.append(n_features)
    mse_values.append(mse)

    # 获取文件名作为图的标题的一部分
    file_title = json_file_path.split('/')[-2]
    plot_titles.append(file_title)

# 绘制mse随n_features变化的曲线
plt.figure(figsize=(10, 5))

colors = ['r', 'g', 'b']  # 为不同的曲线设置不同的颜色
for i in range(len(json_file_paths)):
    plt.plot(n_features_values[i], mse_values[i], marker='o', color=colors[i], label=f'{plot_titles[i]} MSE')

plt.title('MSE vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('MSE')
plt.gca().invert_xaxis()  # 假设随着特征数减少，MSE可能会增加
plt.grid(True)
plt.legend()
combined_title = '_'.join(plot_titles)
plt.savefig(f'{combined_title}.png')  # 保存图像
plt.close()

print(f"Combined plot saved as {combined_title}.png")
