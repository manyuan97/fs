import argparse
import gc
import json
import os

from core.data_helper import DataProcessor
from core.eval_helper import ModelEvaluator
from core.model_helper import ModelTrainer, FeatureSelectorByModel
from core.result_helper import ResultSaver
from core.util_helper import visualize_selected_features


def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_column, alpha, precompute, save_dir,
                   regressor_name, regressor_params):
    model_trainer = ModelTrainer(model_name='lasso', alpha=alpha, precompute=precompute)
    lasso_model = model_trainer.train_model(X_train, y_train)

    feature_selector = FeatureSelectorByModel(lasso_model)
    selector, X_train_selected, selected_features = feature_selector.select_features(X_train, prefit=True)
    visualize_path = os.path.join(save_dir, f'selected_features_alpha_{alpha}.png')
    visualize_selected_features(selected_features, X_train.shape[1], visualize_path)

    with open(regressor_params, 'r') as f:
        params = json.load(f)

    new_model_trainer = ModelTrainer(model_name=regressor_name, params=params)
    regressor = new_model_trainer.train_model(X_train_selected, y_train)

    model_evaluator = ModelEvaluator(regressor)
    metrics_train = model_evaluator.evaluate_model(X_train_selected, y_train)
    metrics_train_positive = model_evaluator.evaluate_model(X_train_selected, y_train, filter_positive_pred=True)

    X_val_selected = selector.transform(X_val)
    metrics_val = model_evaluator.evaluate_model(X_val_selected, y_val)
    metrics_val_positive = model_evaluator.evaluate_model(X_val_selected, y_val, filter_positive_pred=True)

    X_test_selected = selector.transform(X_test)
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
        'selected_features': selected_features.tolist(),
        'n_features': X_test_selected.shape[1],
    }

    del model_trainer, lasso_model, feature_selector, selector, X_train_selected, selected_features, new_model_trainer, regressor, model_evaluator, X_val_selected, X_test_selected
    gc.collect()

    return metrics


def main(target_column, alphas, precompute, save_dir, regressor_name, regressor_params):
    os.makedirs(save_dir, exist_ok=True)

    data_processor = DataProcessor(target_column=target_column)
    train_file_path = 'data/train.parquet'
    val_file_path = 'data/val.parquet'
    test_file_path = 'data/test.parquet'

    # Load data once
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)
    X_val, y_val, _ = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)
    X_test, y_test, _ = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)

    results = {}
    for alpha in alphas:
        print(f"Running experiment with alpha={alpha}")
        metrics = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_column, alpha,
                                 precompute, save_dir, regressor_name, regressor_params)
        results[f"alpha_{alpha}"] = metrics

        json_file_path = os.path.join(save_dir, f"alpha_{alpha}.json")
        result_saver = ResultSaver()
        result_saver.save_metrics_as_json(metrics, json_file_path)
        print(f"Metrics for alpha={alpha} stored in {json_file_path}")

    all_results_file_path = os.path.join(save_dir, "all_alphas.json")
    result_saver.save_metrics_as_json(results, all_results_file_path)
    print("All metrics stored in " + all_results_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, evaluate models and feature selection with alpha tuning.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--precompute', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether to precompute for the Lasso model, true or false.')
    parser.add_argument('--save_dir', type=str, default='./results/lasso_cross',
                        help='Directory to save the results and images.')
    parser.add_argument('--regressor_name', type=str, default='linear_regression',
                        help='The name of the regressor model.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON file containing regressor parameters.')

    args = parser.parse_args()
    alphas = [0.00001, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01,
              0.1]  # Define the range of alpha values to test
    main(args.target_column, alphas, args.precompute, args.save_dir, args.regressor_name, args.regressor_params)
