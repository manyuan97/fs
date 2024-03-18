import argparse
import os

import numpy as np

from core.data_helper import DataProcessor
from core.eval_helper import ModelEvaluator
from core.model_helper import FeatureSelectorByModel
from core.model_helper import ModelTrainer
from core.result_helper import ResultSaver
from core.util_helper import visualize_selected_features, load_params_from_file


def main(target_column, model_name, model_params_path, regressor_name, regressor_params_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    model_params = load_params_from_file(model_params_path)
    regressor_params = load_params_from_file(regressor_params_path)
    train_file_path = 'data/train.parquet'
    val_file_path = 'data/val.parquet'
    test_file_path = 'data/test.parquet'

    data_processor = DataProcessor(target_column=target_column)
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)
    X_val, y_val, _ = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)
    X_test, y_test, _ = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)

    model_trainer = ModelTrainer(model_name=model_name, params=model_params)
    xgboost_model = model_trainer.train_model(X_train, y_train)

    feature_importances = xgboost_model.feature_importances_
    thresholds = np.sort(np.unique(feature_importances))

    split_points = np.linspace(start=thresholds[0], stop=thresholds[-1], num=21)

    selected_thresholds = [split_points[0],split_points[2],split_points[4],split_points[6]]
    for i in range(len(split_points) - 1):
        current_interval_thresholds = thresholds[(thresholds >= split_points[i]) & (thresholds < split_points[i + 1])]
        if len(current_interval_thresholds) > 0:
            selected_thresholds.append(current_interval_thresholds[-1])

    selected_thresholds.append(split_points[-1])

    results = []
    best_mse = float('-inf')
    best_thresh = 0
    best_selector = None
    best_model = None
    best_X_selected = None
    best_selected_feature = None

    for thresh in set(selected_thresholds):

        feature_selector = FeatureSelectorByModel(xgboost_model)
        selector, X_train_selected, _ = feature_selector.select_features(X_train, threshold=thresh, prefit=True)
        select_X_train = selector.transform(X_train)

        regressor_trainer = ModelTrainer(model_name=regressor_name, params=regressor_params)
        regressor = regressor_trainer.train_model(select_X_train, y_train)

        select_X_val = selector.transform(X_val)
        select_X_test = selector.transform(X_test)

        model_evaluator = ModelEvaluator(regressor)
        metrics_train = model_evaluator.evaluate_model(select_X_train, y_train)
        metrics_val = model_evaluator.evaluate_model(select_X_val, y_val)
        metrics_test = model_evaluator.evaluate_model(select_X_test, y_test)

        metrics_train_pos = model_evaluator.evaluate_model(select_X_train, y_train, filter_positive_pred=True)
        metrics_val_pos = model_evaluator.evaluate_model(select_X_val, y_val, filter_positive_pred=True)
        metrics_test_pos = model_evaluator.evaluate_model(select_X_test, y_test, filter_positive_pred=True)

        mse = metrics_val.__dict__['pearson_corr_without_mean']
        print(f"Thresh={thresh:.3f}, n={select_X_train.shape[1]}, pearson_corr_without_mean: {mse:.2f}")

        selected_features = selector.get_support()

        temp_metrics={

            'threshold': thresh,
            'n_features': select_X_train.shape[1],
            'pearson_corr_without_mean': mse,
            'results': {
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__,
            'train_pos': metrics_train_pos.__dict__,
            'val_pos': metrics_val_pos.__dict__,
            'test_pos': metrics_test_pos.__dict__,
            },
            'selected_features': selected_features.tolist(),
        }

        result_saver = ResultSaver()
        json_file_path = os.path.join(save_dir, 'thre_{:.5f}.json'.format(thresh))
        result_saver.save_metrics_as_json(temp_metrics, json_file_path)

        if mse > best_mse:
            best_mse = mse
            best_thresh = thresh
            best_selector = selector
            best_model = regressor
            best_X_selected = select_X_train
            best_selected_feature = selected_features

        visualize_selected_features(selected_features, X_train.shape[1],
                                    os.path.join(save_dir, f'features_{thresh:.3f}.png'))

    model_evaluator = ModelEvaluator(best_model)

    metrics_train = model_evaluator.evaluate_model(best_X_selected, y_train)
    metrics_train_positive = model_evaluator.evaluate_model(best_X_selected, y_train, filter_positive_pred=True)

    X_val_selected = best_selector.transform(X_val)

    metrics_val = model_evaluator.evaluate_model(X_val_selected, y_val)
    metrics_val_positive = model_evaluator.evaluate_model(X_val_selected, y_val, filter_positive_pred=True)

    X_test_selected = best_selector.transform(X_test)

    metrics_test = model_evaluator.evaluate_model(X_test_selected, y_test)
    metrics_test_positive = model_evaluator.evaluate_model(X_test_selected, y_test, filter_positive_pred=True)

    final_metrics = {
        'model_name': model_name,
        'results': {
            'best_threshold': best_thresh,
            'cross_result': results,
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__,
            'train_pos': metrics_train_positive.__dict__,
            'val_pos': metrics_val_positive.__dict__,
            'test_pos': metrics_test_positive.__dict__,

        },
        'args': args_dict,
        'selected_features': best_selected_feature.tolist(),

    }

    result_saver = ResultSaver()
    json_file_path = os.path.join(save_dir, 'results.json')
    result_saver.save_metrics_as_json(final_metrics, json_file_path)

    print(f"Best threshold: {best_thresh}, Validation and Test MSE recorded.")
    print(f"All metrics stored in {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training and evaluation pipeline.")
    parser.add_argument("--target_column", type=str, default='y1', help="Name of the target column.")
    parser.add_argument("--model_name", type=str, default="xgboost", help="Name of the initial model.")
    parser.add_argument("--save_dir", type=str, default='results/xgboost_cross', help="Directory to save the results.")
    parser.add_argument('--regressor_name', type=str, default='linear_regression',
                        help='The target column for prediction.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON string of regressor parameters.')
    parser.add_argument('--model_params', type=str, default='./configs/xgboost.json',
                        help='File path to JSON string of model parameters.')

    args = parser.parse_args()
    args_dict = vars((args))
    main(args.target_column, args.model_name, args.model_params, args.regressor_name, args.regressor_params,
         args.save_dir)
