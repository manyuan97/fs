import pandas as pd
from sklearn.datasets import make_regression

# 生成训练数据集
X_train, y_train = make_regression(n_samples=1000, n_features=1246, n_targets=3, random_state=42)

# 生成验证数据集
X_val, y_val = make_regression(n_samples=100, n_features=1246, n_targets=3, random_state=42)

# 生成测试数据集
X_test, y_test = make_regression(n_samples=100, n_features=1246, n_targets=3, random_state=42)

# 将训练数据保存为 Parquet 文件
train_df = pd.DataFrame(X_train, columns=[f'x{i}' for i in range(X_train.shape[1])])
train_df['y1'] = y_train[:, 0]
train_df['y2'] = y_train[:, 1]
train_df['y3'] = y_train[:, 2]
train_df.to_parquet('./train.parquet')

# 将验证数据保存为 Parquet 文件
val_df = pd.DataFrame(X_val, columns=[f'x{i}' for i in range(X_val.shape[1])])
val_df['y1'] = y_val[:, 0]
val_df['y2'] = y_val[:, 1]
val_df['y3'] = y_val[:, 2]
val_df.to_parquet('./val.parquet')

# 将测试数据保存为 Parquet 文件
test_df = pd.DataFrame(X_test, columns=[f'x{i}' for i in range(X_test.shape[1])])
test_df['y1'] = y_test[:, 0]
test_df['y2'] = y_test[:, 1]
test_df['y3'] = y_test[:, 2]
test_df.to_parquet('./test.parquet')