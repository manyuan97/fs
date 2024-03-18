import json
import os
from datetime import datetime, timedelta

# 指定结果文件夹的路径
results_folder = './results'

# 遍历results文件夹下的每个子文件夹
# for subdir, dirs, files in os.walk(results_folder):
#     for file in files:
#         # 检查文件是否是JSON文件
#         if file.endswith('.json'):
#             # 构造完整的文件路径
#             json_file_path = os.path.join(subdir, file)
#             # 获取文件的最后修改时间
#             last_modified_time = os.path.getmtime(json_file_path)
#             # 将时间戳转换为人类可读的时间格式
#             readable_time = datetime.fromtimestamp(last_modified_time).strftime('%Y-%m-%d %H:%M:%S')
#             print(f"File: {file} was last modified at {readable_time}")

now = datetime.now()

# 遍历results文件夹下的每个子文件夹
for subdir, dirs, files in os.walk(results_folder):
    for file in files:
        # 检查文件是否是JSON文件
        if file.endswith('.json'):
            # 构造完整的文件路径
            json_file_path = os.path.join(subdir, file)
            # 获取文件的最后修改时间
            last_modified_time = os.path.getmtime(json_file_path)
            # 计算当前时间与文件最后修改时间的差值
            time_difference = now - datetime.fromtimestamp(last_modified_time)
            # 检查差值是否大于48小时
            if time_difference > timedelta(hours=48):
                readable_time = datetime.fromtimestamp(last_modified_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"File: {json_file_path} was last modified more than 48 hours ago at {readable_time}")