import json
import matplotlib.pyplot as plt

# 替换为你的JSON文件的路径
json_file_path = 'results/xgboost_LR_y1_cross/results.json'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 提取cross_result中的n_features和mse
n_features = [item['n_features'] for item in data['results']['cross_result']]
mse = [item['mse'] for item in data['results']['cross_result']]

# 获取文件名作为图的标题
file_title = json_file_path.split('/')[-2].split('.')[0]

# 绘制mse随n_features变化的曲线
plt.figure(figsize=(10, 5))
plt.plot(n_features, mse, marker='o')
plt.title(f'MSE vs Number of Features for {file_title}')
plt.xlabel('Number of Features')
plt.ylabel('MSE')
plt.gca().invert_xaxis()  # 假设随着特征数减少，MSE可能会增加
plt.grid(True)
plt.savefig(f'{file_title}.png')  # 保存图像
plt.close()

print(f"Plot saved as {file_title}.png")