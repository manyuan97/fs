
from sklearn.preprocessing import StandardScaler
import pandas as pd
from util_helper import Utils

@Utils.timeit
class DataProcessor:
    def __init__(self, target_column='y1'):
        self.target_column = target_column

    def load_and_preprocess_data(self, file_path, scaler=None):
        df = pd.read_parquet(file_path)
        if self.target_column == 'y1':
            df = df.drop(['y2', 'y3'], axis=1)
        elif self.target_column == 'y2':
            df = df.drop(['y1', 'y3'], axis=1)
        elif self.target_column == 'y3':
            df = df.drop(['y1', 'y2'], axis=1)
        else:
            raise ValueError("Invalid target_column value. It should be 'y1', 'y2', or 'y3'.")

        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, y, scaler