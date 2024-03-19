import pandas as pd

file_path = '/cpfs/user2/mzhang2/full_data/train.parquet'
df = pd.read_parquet(file_path)

# 10%
df_10_percent = df.sample(frac=0.1, random_state=42)
new_file_path_10 = '/cpfs/user2/mzhang2/split_data/10_percent.parquet'
df_10_percent.to_parquet(new_file_path_10)
print(f"Saved 10% of the data to: {new_file_path_10}")

# 20%
df_20_percent = df.sample(frac=0.2, random_state=42)
new_file_path_20 = '/cpfs/user2/mzhang2/split_data/20_percent.parquet'
df_20_percent.to_parquet(new_file_path_20)
print(f"Saved 20% of the data to: {new_file_path_20}")

# 30%
df_30_percent = df.sample(frac=0.3, random_state=42)
new_file_path_30 = '/cpfs/user2/mzhang2/split_data/30_percent.parquet'
df_30_percent.to_parquet(new_file_path_30)
print(f"Saved 30% of the data to: {new_file_path_30}")

# 50%
df_50_percent = df.sample(frac=0.5, random_state=42)
new_file_path_50 = '/cpfs/user2/mzhang2/split_data/50_percent.parquet'
df_50_percent.to_parquet(new_file_path_50)
print(f"Saved 50% of the data to: {new_file_path_50}")

# 75%
df_75_percent = df.sample(frac=0.75, random_state=42)
new_file_path_75 = '/cpfs/user2/mzhang2/split_data/75_percent.parquet'
df_75_percent.to_parquet(new_file_path_75)
print(f"Saved 75% of the data to: {new_file_path_75}")