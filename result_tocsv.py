import json
import os
import csv

# 指定结果文件夹的路径
results_folder = './results'

# CSV文件的路径
csv_file_path = './results_summary.csv'

# 创建CSV文件并写入标题行
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    # 假设所有JSON文件结构相同，标题行可以根据JSON键来设置
    headers = ['model_name', 'train_mse', 'train_pearson_corr', 'train_mean_value', 'train_std_dev', 'train_skewness', 'train_kurt', 'train_positive_count', 'train_negative_count',
               'val_mse', 'val_pearson_corr', 'val_mean_value', 'val_std_dev', 'val_skewness', 'val_kurt', 'val_positive_count', 'val_negative_count',
               'test_mse', 'test_pearson_corr', 'test_mean_value', 'test_std_dev', 'test_skewness', 'test_kurt', 'test_positive_count', 'test_negative_count']
    csv_writer.writerow(headers)

    # 遍历results文件夹下的每个子文件夹
    for subdir, dirs, files in os.walk(results_folder):
        for file in files:
            # 检查文件是否是JSON文件
            if file.endswith('.json'):
                # 构造完整的文件路径
                json_file_path = os.path.join(subdir, file)
                # 读取JSON文件
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    # 提取你想要的数据
                    model_name = data.get('model_name', '')
                    # 初始化行数据列表
                    row = [json_file_path]
                    # 遍历每个数据集（train, val, test）
                    for dataset in ('train', 'val', 'test'):
                        results = data['results'].get(dataset, {})
                        # 添加数据集的每个指标
                        row.extend([
                            results.get('mse', ''),
                            results.get('pearson_corr', ''),
                            results.get('mean_value', ''),
                            results.get('std_dev', ''),
                            results.get('skewness', ''),
                            results.get('kurt', ''),
                            results.get('positive_count', ''),
                            results.get('negative_count', '')
                        ])
                    # 写入数据行到CSV
                    csv_writer.writerow(row)

print(f"CSV summary file created at: {csv_file_path}")