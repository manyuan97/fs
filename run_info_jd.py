import argparse
import gc
import json
import numpy as np
import os

# 假设以下类已经根据之前的描述被正确实现
from data_helper import DataProcessor
from eval_helper import ModelEvaluator
from model_helper import ModelTrainer
from result_helper import ResultSaver
from util_helper import visualize_selected_features, load_params_from_file
from sklearn.feature_selection import mutual_info_regression, f_regression


# 特征选择类也需要被实现
from model_helper import * # 假设FeatureSelectorByCIFE已经被实现


def main(target_column, k, feature_selection_method, regressor_name, regressor_params_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data_processor = DataProcessor(target_column=target_column)

    regressor_params = load_params_from_file(regressor_params_path)

    train_file_path = './train.parquet'
    val_file_path = './val.parquet'
    test_file_path = './test.parquet'

    # 加载和预处理数据
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)

    if feature_selection_method == 'mutual_info_regression':
        feature_selector = FeatureSelectorByK(mutual_info_regression, k=k)
    elif feature_selection_method == 'f_regression':
        feature_selector = FeatureSelectorByK(f_regression, k=k)
    elif feature_selection_method == 'cife':
        feature_selector = FeatureSelectorByCIFE(n_selected_features=k)
    elif feature_selection_method == 'fcbf':
        feature_selector = FeatureSelectorByFCBF(delta=0.0, n_selected_features=k)
    elif feature_selection_method == 'lcsi':
        feature_selector = FeatureSelectorByLCSI(beta=1.0, gamma=1.0, n_selected_features=k)
    elif feature_selection_method == 'mrmr':
        feature_selector = FeatureSelectorByMRMR(n_selected_features=k)
    elif feature_selection_method == 'cmim':
        feature_selector = FeatureSelectorByCMIM(n_selected_features=k)
    elif feature_selection_method == 'icap':
        feature_selector = FeatureSelectorByICAP(n_selected_features=k)
    elif feature_selection_method == 'mifs':
        feature_selector = FeatureSelectorByMIFS(n_selected_features=k)
    elif feature_selection_method == 'disr':
        feature_selector = FeatureSelectorByDISR(n_selected_features=k)
    elif feature_selection_method == 'jmi':
        feature_selector = FeatureSelectorByJMI(n_selected_features=k)
    elif feature_selection_method == 'mim':
        feature_selector = FeatureSelectorByMIM(n_selected_features=k)
    else:
        raise ValueError("Unsupported feature selection method")

    _, X_train_selected, selected_features = feature_selector.select_features(X_train, y_train)

    visualize_path = os.path.join(save_dir, 'selected_features.png')
    visualize_selected_features(selected_features, X_train.shape[1], visualize_path)

    model_trainer = ModelTrainer(model_name=regressor_name, params=regressor_params)
    regressor = model_trainer.train_model(X_train_selected, y_train)

    model_evaluator = ModelEvaluator(regressor)

    metrics_train = model_evaluator.evaluate_model(X_train_selected, y_train)
    metrics_train_positive = model_evaluator.evaluate_model(X_train_selected, y_train, filter_positive_pred=True)

    del X_train, y_train, X_train_selected
    gc.collect()

    X_val, y_val, _ = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)

    X_val_selected = feature_selector.transform(X_val)
    metrics_val = model_evaluator.evaluate_model(X_val_selected, y_val)
    metrics_val_positive = model_evaluator.evaluate_model(X_val_selected, y_val, filter_positive_pred=True)

    del X_val, y_val, X_val_selected
    gc.collect()

    X_test, y_test, _ = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)

    X_test_selected = feature_selector.transform(X_test)
    metrics_test = model_evaluator.evaluate_model(X_test_selected, y_test)
    metrics_test_positive = model_evaluator.evaluate_model(X_test_selected, y_test, filter_positive_pred=True)

    metrics = {
        'model_name': regressor_name,
        'results': {
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__,
            'train_pos': metrics_train_positive.__dict__,
            'val_pos': metrics_val_positive.__dict__,
            'test_pos': metrics_test_positive.__dict__,
        },
        'args': args_dict,
        'selected_features': selected_features.tolist(),
    }

    result_saver = ResultSaver()
    json_file_path = os.path.join(save_dir, 'results.json')
    result_saver.save_metrics_as_json(metrics, json_file_path)

    print(f"Metrics stored in {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models with feature selection.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--k', type=int, default=100, help='The number of top features to select.')
    # parser.add_argument('--feature_selection', type=str, default='f_regression',
    #                     choices=['mutual_info_regression', 'f_regression'], help='The feature selection method to use.')
    # parser.add_argument('--save_dir', type=str, default='results/f_regression',
    #                     help='The file path where the metrics JSON will be saved.')
    parser.add_argument('--feature_selection', type=str, default='mim',
                        choices=[
                            'mutual_info_regression', 'f_regression', 'cife',
                            'fcbf', 'lcsi', 'mrmr', 'cmim', 'icap', 'mifs',
                            'disr', 'jmi', 'mim'
                        ],
                        help='The feature selection method to use.')
    parser.add_argument('--save_dir', type=str, default='results/mutual_info_regression',
                        help='The file path where the metrics JSON will be saved.')
    parser.add_argument('--regressor_name', type=str, default='linear_regression',
                        help='The name of the regressor model.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON file containing regressor parameters.')

    args = parser.parse_args()
    args_dict = vars(args)

    main(args.target_column, args.k, args.feature_selection, args.regressor_name, args.regressor_params, args.save_dir)
