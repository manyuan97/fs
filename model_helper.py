from sklearn.feature_selection import SelectFromModel
from util_helper import Utils
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
import numpy as np

from sklearn.feature_selection import SelectKBest
from skfeature.function.information_theoretical_based.CIFE import cife
from skfeature.function.information_theoretical_based.FCBF import fcbf
from skfeature.function.information_theoretical_based.LCSI import lcsi
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.CMIM import cmim
from skfeature.function.information_theoretical_based.ICAP import icap
from skfeature.function.information_theoretical_based.MIFS import mifs
from skfeature.function.information_theoretical_based.DISR import disr
from skfeature.function.information_theoretical_based.JMI import jmi
from skfeature.function.information_theoretical_based.MIM import mim



class FeatureSelectorByMIM:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用MIM选择特征
        F = mim(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征

class FeatureSelectorByJMI:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用JMI选择特征
        F = jmi(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征




class FeatureSelectorByCIFE:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用CIFE选择特征
        F = cife(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征


class FeatureSelectorByFCBF:
    def __init__(self, delta=0, n_selected_features=100):
        self.delta = delta
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用FCBF选择特征
        F = fcbf(X_train, y_train, delta=self.delta)
        self.selected_indices = F  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征


class FeatureSelectorByCMIM:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用CMIM选择特征
        F = cmim(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征



class FeatureSelectorByLCSI:
    def __init__(self, beta=1.0, gamma=1.0, n_selected_features=100):
        self.beta = beta
        self.gamma = gamma
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用LCSI选择特征
        F = lcsi(X_train, y_train, beta=self.beta, gamma=self.gamma, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征


class FeatureSelectorByMRMR:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用MRMR选择特征
        F = mrmr(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征


class FeatureSelectorByICAP:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用ICAP选择特征
        F = icap(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征

class FeatureSelectorByMIFS:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用MIFS选择特征
        F = mifs(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征

class FeatureSelectorByDISR:
    def __init__(self, n_selected_features=100):
        self.n_selected_features = n_selected_features
        self.selected_indices = None  # 用于存储被选特征的索引

    @Utils.timeit
    def select_features(self, X_train, y_train):
        # 使用DISR选择特征
        F = disr(X_train, y_train, n_selected_features=self.n_selected_features)
        self.selected_indices = F[0]  # 存储被选特征的索引
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[F[0]] = True  # 标记被选特征为True
        X_train_selected = X_train[:, F[0]]  # 选取被选特征
        return None, X_train_selected, selected_features

    def transform(self, X):
        # 检查是否已进行了特征选择
        if self.selected_indices is None:
            raise ValueError("The select_features method must be called before transform.")
        return X[:, self.selected_indices]  # 返回选取的特征

class FeatureSelectorByK:
    def __init__(self, score_func, k=100):

        self.score_func = score_func
        self.k = k

    @Utils.timeit
    def select_features(self, X_train, y_train):
        selector = SelectKBest(self.score_func, k=self.k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support()  # This will be a boolean array
        return selector, X_train_selected, selected_features


class FeatureSelectorByMethod:
    def __init__(self, method='variance_threshold', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.selector = None

    @Utils.timeit
    def apply_feature_selection(self, X_train):
        if self.method == 'variance':
            self.selector = VarianceThreshold(threshold=self.kwargs.get('threshold', 0.0))
        elif self.method == 'pca':
            self.selector = PCA(n_components=self.kwargs.get('n_components', 0.95))
        elif self.method == 'ica':
            self.selector = FastICA(n_components=self.kwargs.get('n_components', 10),
                                    random_state=self.kwargs.get('random_state', 0))
        else:
            raise ValueError("Unsupported feature selection method")

        X_transformed = self.selector.fit_transform(X_train)
        print(
            f"Method: {self.method}, Original feature count: {X_train.shape[1]}, New feature count: {X_transformed.shape[1]}")
        return self.selector, X_transformed

class FeatureSelectorByModel:
    def __init__(self, model):
        self.model = model

    @Utils.timeit
    def select_features(self, X_train, **kwargs):
        feature_selector = SelectFromModel(self.model, **kwargs)
        X_train_selected = feature_selector.transform(X_train)
        selected_features = feature_selector.get_support()
        return feature_selector, X_train_selected, selected_features


class ModelTrainer:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.model_kwargs = kwargs

    @Utils.timeit
    def train_model(self, X_train, y_train):
        if self.model_name == 'lasso':
            alpha = self.model_kwargs.get('alpha', 0.01)
            precompute = self.model_kwargs.get('precompute', True)
            model = Lasso(alpha=alpha, precompute=precompute)
        elif self.model_name == 'lightgbm':
            params = self.model_kwargs.get('params', {})
            model = LGBMRegressor(**params)
        elif self.model_name == 'xgboost':
            params = self.model_kwargs.get('params', {})
            model = XGBRegressor(**params)
        elif self.model_name == 'linear_regression':
            model = LinearRegression()
        elif self.model_name == 'linear_svr':
            model = LinearSVR(C=self.model_kwargs.get('C', 0.01), max_iter=self.model_kwargs.get('max_iter', 10000))
        elif self.model_name == 'decision_tree':
            model = DecisionTreeRegressor(random_state=self.model_kwargs.get('random_state', 42))
        elif self.model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=self.model_kwargs.get('n_estimators', 100),
                                              random_state=self.model_kwargs.get('random_state', 42))
        elif self.model_name == 'adaboost':
            model = AdaBoostRegressor(n_estimators=self.model_kwargs.get('n_estimators', 100),
                                      random_state=self.model_kwargs.get('random_state', 42))
        elif self.model_name == 'random_forest':
            model = RandomForestRegressor(n_estimators=self.model_kwargs.get('n_estimators', 100),
                                          random_state=self.model_kwargs.get('random_state', 42))
        else:
            raise ValueError("Unsupported model type")

        model.fit(X_train, y_train)
        return model