import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from .util_helper import Utils

class EvaluationResult:
    def __init__(self, mse, pearson_corr, pearson_corr_without_mean, conf_matrix, mean_value, std_dev, skewness, kurt,
                 vol, mean_over_vol, pos_ratio, neg_ratio, pos_neg_ratio):
        self.mse = mse
        self.pearson_corr = pearson_corr
        self.pearson_corr_without_mean = pearson_corr_without_mean
        self.conf_matrix = conf_matrix
        self.mean_value = mean_value
        self.std_dev = std_dev
        self.skewness = skewness
        self.kurt = kurt
        self.vol = vol
        self.mean_over_vol = mean_over_vol
        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        self.pos_neg_ratio = pos_neg_ratio


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def pearson_corr_without_mean(self, y_true, y_pred):
        sum_xy = np.mean(y_true * y_pred)
        sum_x2 = np.mean(y_true * y_true)
        sum_y2 = np.mean(y_pred * y_pred)
        return sum_xy / np.sqrt(sum_x2 * sum_y2)

    def stats_eval(self, y_pred, y_val):
        pearson_corr, _ = pearsonr(y_val, y_pred)
        y_val_binary = (y_val > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        conf_matrix = confusion_matrix(y_val_binary, y_pred_binary)
        conf_matrix_ratio = conf_matrix / np.sum(conf_matrix)
        pearson_corr_without_mean = self.pearson_corr_without_mean(y_val, y_pred)
        return pearson_corr, conf_matrix_ratio, pearson_corr_without_mean

    def stat_single(self, y_pred):
        mean_value = np.mean(y_pred)
        std_dev = np.std(y_pred)
        skewness = skew(y_pred)
        kurt = kurtosis(y_pred)
        vol = np.sqrt(np.mean(np.square(y_pred)))
        mean_over_vol = mean_value / vol if vol != 0 else np.inf
        positive_ratio = np.mean(y_pred > 0)
        negative_ratio = np.mean(y_pred < 0)
        pos_neg_ratio = positive_ratio / negative_ratio if negative_ratio != 0 else np.inf
        return mean_value, std_dev, skewness, kurt, vol, mean_over_vol, positive_ratio, negative_ratio, pos_neg_ratio

    @Utils.timeit
    def evaluate_model(self, X_val, y_val,filter_positive_pred=False):
        y_pred = self.model.predict(X_val)

        if filter_positive_pred:
            y_val = y_val[y_pred > 0]
            y_pred = y_pred[y_pred > 0]

        pearson_corr, conf_matrix_ratio, pearson_corr_without_mean = self.stats_eval(y_pred, y_val)
        mean_value, std_dev, skewness, kurt, vol, mean_over_vol, pos_ratio, neg_ratio, pos_neg_ratio = self.stat_single(
            y_pred)
        mse = mean_squared_error(y_val, y_pred)
        return EvaluationResult(mse, pearson_corr, pearson_corr_without_mean, conf_matrix_ratio, mean_value, std_dev,
                                skewness, kurt, vol, mean_over_vol, pos_ratio, neg_ratio, pos_neg_ratio)
