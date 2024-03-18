import csv

# CSV文件的路径
csv_file_path = './results_summary.csv'

# 读取CSV文件到内存
data_rows = []
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row['dataset_type'] == 'test':
            # 只考虑dataset_type为test的行
            data_rows.append(row)

# 根据pearson_corr_without_mean排序，注意将值转换为浮点数进行比较
sorted_rows = sorted(data_rows, key=lambda x: float(x['pearson_corr_without_mean']) if x['pearson_corr_without_mean'] else 0, reverse=True)

# 打印前十行
for i, row in enumerate(sorted_rows[:10]):
    print(f"Row {i+1}: {row}")