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
    # 根据新的数据结构设置标题行
    headers = ['model_name', 'dataset_type', 'mse', 'pearson_corr', 'pearson_corr_without_mean', 'conf_matrix', 'mean_value',
               'std_dev', 'skewness', 'kurt', 'vol', 'mean_over_vol', 'pos_ratio', 'neg_ratio', 'pos_neg_ratio']
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
                    # 提取模型名称
                    model_name = data.get('model_name', '')
                    # 遍历每个数据集类型（train, val, test, train_pos, val_pos, test_pos）
                    for dataset_type in ('train', 'val', 'test', 'train_pos', 'val_pos', 'test_pos'):
                        results = data['results'].get(dataset_type, {})
                        # 写入数据行到CSV
                        csv_writer.writerow([
                            json_file_path,
                            dataset_type,
                            results.get('mse', ''),
                            results.get('pearson_corr', ''),
                            results.get('pearson_corr_without_mean', ''),
                            str(results.get('conf_matrix', '')),  # 因为conf_matrix是一个结构体或列表，所以转换成字符串
                            results.get('mean_value', ''),
                            results.get('std_dev', ''),
                            results.get('skewness', ''),
                            results.get('kurt', ''),
                            results.get('vol', ''),
                            results.get('mean_over_vol', ''),
                            results.get('pos_ratio', ''),
                            results.get('neg_ratio', ''),
                            results.get('pos_neg_ratio', '')
                        ])

print(f"CSV summary file created at: {csv_file_path}")