import argparse
import gc
import os

from sklearn.feature_selection import mutual_info_regression, f_regression

from core.data_helper import DataProcessor
from core.eval_helper import ModelEvaluator
from core.model_helper import ModelTrainer, FeatureSelectorByK
from core.result_helper import ResultSaver
from core.util_helper import visualize_selected_features, load_params_from_file


def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_column, k, feature_selection_method,
                   regressor_name, regressor_params, save_dir):
    if feature_selection_method == 'mutual_info_regression':
        feature_selector = FeatureSelectorByK(mutual_info_regression, k=k)
    elif feature_selection_method == 'f_regression':
        feature_selector = FeatureSelectorByK(f_regression, k=k)
    else:
        raise ValueError("Unsupported feature selection method")

    selector, X_train_selected, selected_features = feature_selector.select_features(X_train, y_train)

    visualize_path = os.path.join(save_dir, f'selected_features_k_{k}.png')
    visualize_selected_features(selected_features, X_train.shape[1], visualize_path)

    model_trainer = ModelTrainer(model_name=regressor_name, params=regressor_params)
    regressor = model_trainer.train_model(X_train_selected, y_train)

    model_evaluator = ModelEvaluator(regressor)
    metrics = {
        'results': {
            'train': model_evaluator.evaluate_model(X_train_selected, y_train).__dict__,
            'val': model_evaluator.evaluate_model(selector.transform(X_val), y_val).__dict__,
            'test': model_evaluator.evaluate_model(selector.transform(X_test), y_test).__dict__,

            'train_pos': model_evaluator.evaluate_model(X_train_selected, y_train, filter_positive_pred=True).__dict__,
            'val_pos': model_evaluator.evaluate_model(selector.transform(X_val), y_val,
                                                      filter_positive_pred=True).__dict__,
            'test_pos': model_evaluator.evaluate_model(selector.transform(X_test), y_test,
                                                       filter_positive_pred=True).__dict__,
        },
        'selected_features': selected_features.tolist(),
        'k': k
    }

    del selector, X_train_selected, selected_features, model_trainer, regressor, model_evaluator
    gc.collect()

    return metrics


def main(target_column, ks, feature_selection_method, regressor_name, regressor_params_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data_processor = DataProcessor(target_column=target_column)
    regressor_params = load_params_from_file(regressor_params_path)

    train_file_path = 'data/train.parquet'
    val_file_path = 'data/val.parquet'
    test_file_path = 'data/test.parquet'

    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)
    X_val, y_val, _ = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)
    X_test, y_test, _ = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)

    results = {}
    for k in ks:
        print(f"Running experiment with k={k}")
        metrics = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_column, k,
                                 feature_selection_method, regressor_name, regressor_params, save_dir)
        results[f"k_{k}"] = metrics

        json_file_path = os.path.join(save_dir, f"k_{k}_results.json")
        result_saver = ResultSaver()
        result_saver.save_metrics_as_json(metrics, json_file_path)
        print(f"Metrics for k={k} stored in {json_file_path}")

    all_results_file_path = os.path.join(save_dir, "all_ks_results.json")
    result_saver.save_metrics_as_json(results, all_results_file_path)
    print("All metrics stored in " + all_results_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate models with feature selection based on k top features.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--feature_selection', type=str, default='mutual_info_regression',
                        choices=['mutual_info_regression', 'f_regression'], help='The feature selection method to use.')
    parser.add_argument('--save_dir', type=str, default='results/mutual_info_regression',
                        help='The file path where the metrics JSON will be saved.')
    parser.add_argument('--regressor_name', type=str, default='linear_regression',
                        help='The name of the regressor model.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON file containing regressor parameters.')

    args = parser.parse_args()

    ks = [10, 20, 50, 100, 200, 400, 500, 800, 1000]
    main(args.target_column, ks, args.feature_selection, args.regressor_name, args.regressor_params, args.save_dir)
