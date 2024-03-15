import json
import numpy as np
import matplotlib.pyplot as plt
import math
import sys



# JSON文件路径来自命令行参数
json_file_paths = ['./results/xgboost_xgboost_y1/results.json',
                   './results/xgboost_xgboost_y2/results.json',
                   './results/xgboost_xgboost_y3/results.json']

# 累积selected_features的次数
selected_features_counts = None

# 遍历每个JSON文件路径
for json_file_path in json_file_paths:
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        selected_features = np.array(data['selected_features'])
        # 累积selected_features
        if selected_features_counts is None:
            selected_features_counts = selected_features
        else:
            # 确保所有特征列表长度一致
            min_length = min(len(selected_features_counts), len(selected_features))
            selected_features_counts[:min_length] += selected_features[:min_length]

# 定义可视化函数
def visualize_selected_features_across_models(selected_features_counts, filename):
    num_features = len(selected_features_counts)
    dim = math.ceil(np.sqrt(num_features))
    image = np.zeros((dim ** 2,))
    image[:len(selected_features_counts)] = selected_features_counts
    image = image.reshape((dim, dim))

    # 为每种选择次数设置不同的颜色深度
    # 没有被选择的特征（0次）为白色，1次为浅色，2次为中等色，3次为深色
    cmap = plt.cm.get_cmap('Greys', 4)  # 创建一个有4个颜色的色阶
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Plot and save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title('Accumulated Selected Features Visualization')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Number of models that selected the feature')
    plt.axis('off')  # Hide the axes
    plt.savefig(filename)
    plt.close()

# 生成累积特征选择的可视化图像
visualize_filename = './accumulated_selected_features_visualization.png'
visualize_selected_features_across_models(selected_features_counts, visualize_filename)
print(f"Accumulated selected features visualization saved to: {visualize_filename}")
