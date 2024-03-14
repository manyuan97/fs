import argparse
import gc
import os

import matplotlib.pyplot as plt
import numpy as np

from data_helper import DataProcessor
from eval_helper import ModelEvaluator
from model_helper import ModelTrainer, FeatureSelectorByModel
from result_helper import ResultSaver


def visualize_selected_features(selected_features, num_features, filename):
    """
    Visualize the selected features as a square with highlighted (selected) areas.
    :param selected_features: Boolean array indicating selected features.
    :param num_features: Total number of features.
    :param filename: Filename to save the visualization.
    """
    dim = int(np.ceil(np.sqrt(num_features)))
    image = np.zeros((dim ** 2,))
    image[:len(selected_features)] = selected_features.astype(int)
    image = image.reshape((dim, dim))

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys', interpolation='nearest')
    plt.title('Selected Features Visualization')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def run_experiment(target_column, alpha, precompute, save_dir):
    train_file_path = './train.parquet'
    val_file_path = './val.parquet'
    test_file_path = './test.parquet'

    data_processor = DataProcessor(target_column=target_column)
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)
    model_trainer = ModelTrainer(model_name='lasso', alpha=alpha, precompute=precompute)
    lasso_model = model_trainer.train_model(X_train, y_train)

    feature_selector = FeatureSelectorByModel(lasso_model)
    _, X_train_selected, selected_features = feature_selector.select_features(X_train, prefit=True)
    visualize_path = os.path.join(save_dir, f'selected_features_alpha_{alpha}.png')
    visualize_selected_features(selected_features, X_train.shape[1], visualize_path)

    model_evaluator = ModelEvaluator(lasso_model)
    metrics_train = model_evaluator.evaluate_model(X_train, y_train)
    del X_train, y_train, X_train_selected
    gc.collect()

    X_val, y_val, scaler = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)
    metrics_val = model_evaluator.evaluate_model(X_val, y_val)
    del X_val, y_val
    gc.collect()

    X_test, y_test, scaler = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)
    metrics_test = model_evaluator.evaluate_model(X_test, y_test)

    metrics = {
        'model_name': 'lasso',
        'results': {
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__},
    }

    return metrics


def main(target_column, alphas, precompute, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    for alpha in alphas:
        json_file_path = os.path.join(save_dir, f"alpha_{alpha}.json")
        print(f"Running experiment with alpha={alpha}")
        metrics = run_experiment(target_column, alpha, precompute, save_dir)
        results[f"alpha_{alpha}"] = metrics

        result_saver = ResultSaver()
        result_saver.save_metrics_as_json(metrics, json_file_path)
        print(f"Metrics for alpha={alpha} stored in {json_file_path}")

    all_results_file_path = os.path.join(save_dir, f"all_alphas.json")
    result_saver = ResultSaver()
    result_saver.save_metrics_as_json(results, all_results_file_path)
    print(f"All metrics stored in {all_results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, evaluate models and feature selection with alpha tuning.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--precompute', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Whether to precompute for the Lasso model, true or false.')
    parser.add_argument('--save_dir', type=str, default='./results/lasso_cross',
                        help='Directory to save the results and images.')

    args = parser.parse_args()
    alphas = [0.001, 0.01, 0.1]  # Define the range of alpha values to test
    main(args.target_column, alphas, args.precompute, args.save_dir)
