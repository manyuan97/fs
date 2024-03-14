from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.metrics import confusion_matrix
from util_helper import  Utils

class EvaluationResult:
    def __init__(self, mse, pearson_corr, pearson_corr_without_mean,conf_matrix, mean_value, std_dev, skewness, kurt, positive_count, negative_count):
        self.mse = mse
        self.pearson_corr = pearson_corr
        self.pearson_corr_without_mean = pearson_corr_without_mean
        self.conf_matrix = conf_matrix
        self.mean_value = mean_value
        self.std_dev = std_dev
        self.skewness = skewness
        self.kurt = kurt
        self.positive_count = positive_count
        self.negative_count = negative_count

class ModelEvaluator:
    def __init__(self, model):
        self.model = model


    def pearson_corr_without_mean(self, y_true, y_pred):
        sum_xy = np.sum(y_true * y_pred)
        sum_x2 = np.sum(y_true * y_true)
        sum_y2 = np.sum(y_pred * y_pred)
        return sum_xy / np.sqrt(sum_x2 * sum_y2)

    def stats_eval(self, y_pred, y_val):
        pearson_corr, _ = pearsonr(y_val, y_pred)
        y_val_binary = (y_val > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        conf_matrix = confusion_matrix(y_val_binary, y_pred_binary)
        pearson_corr_without_mean = self.pearson_corr_without_mean(y_val,y_pred)
        return pearson_corr, conf_matrix, pearson_corr_without_mean


    def stat_single(self, y_pred):
        mean_value = np.mean(y_pred)
        std_dev = np.std(y_pred)
        skewness = skew(y_pred)
        kurt = kurtosis(y_pred)
        positive_count = np.sum(y_pred > 0)
        negative_count = np.sum(y_pred < 0)
        return mean_value, std_dev, skewness, kurt, positive_count, negative_count

    @Utils.timeit
    def evaluate_model(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        pearson_corr, conf_matrix, pearson_corr_without_mean = self.stats_eval(y_pred, y_val)
        mean_value, std_dev, skewness, kurt, positive_count, negative_count = self.stat_single(y_pred)
        mse = mean_squared_error(y_val, y_pred)
        return EvaluationResult(mse, pearson_corr,pearson_corr_without_mean, conf_matrix, mean_value, std_dev, skewness, kurt, positive_count, negative_count)
