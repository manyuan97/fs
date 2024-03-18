import argparse
import gc
import os

from core.data_helper import DataProcessor
from core.eval_helper import ModelEvaluator
from core.model_helper import ModelTrainer, FeatureSelectorByMethod
from core.result_helper import ResultSaver
from core.util_helper import load_params_from_file



def main(target_column, method, method_params_path, regressor_name, regressor_params_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data_processor = DataProcessor(target_column=target_column)

    method_params = load_params_from_file(method_params_path)
    regressor_params = load_params_from_file(regressor_params_path)

    train_file_path = 'data/train.parquet'
    val_file_path = 'data/val.parquet'
    test_file_path = 'data/test.parquet'


    # 加载和预处理数据
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)
    feature_selector = FeatureSelectorByMethod(method=method, **method_params)
    selector, X_train_selected = feature_selector.apply_feature_selection(X_train)

    model_trainer = ModelTrainer(model_name=regressor_name, params=regressor_params)

    regressor = model_trainer.train_model(X_train_selected, y_train)

    model_evaluator = ModelEvaluator(regressor)
    metrics_train = model_evaluator.evaluate_model(X_train_selected, y_train)

    del X_train, y_train, X_train_selected
    gc.collect()

    X_val, y_val, scaler = data_processor.load_and_preprocess_data(val_file_path, scaler=scaler)
    X_val_selected = selector.transform(X_val)
    metrics_val = model_evaluator.evaluate_model(X_val_selected, y_val)

    del X_val, y_val, X_val_selected
    gc.collect()

    X_test, y_test, scaler = data_processor.load_and_preprocess_data(test_file_path, scaler=scaler)

    X_test_selected = selector.transform(X_test)

    metrics_test = model_evaluator.evaluate_model(X_test_selected, y_test)

    # 保存评估结果
    metrics = {
        'model_name': method,
        'results': {
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__
        },
        'args': args_dict
    }

    result_saver = ResultSaver()
    json_file_path = os.path.join(save_dir, 'results.json')
    result_saver.save_metrics_as_json(metrics, json_file_path)

    print(f"Metrics stored in {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models with feature selection.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--method', type=str, default='variance', help='Feature selection method to use.')
    parser.add_argument('--method_params', type=str, default='./configs/variance.json',
                        help='File path to JSON file containing method parameters.')
    parser.add_argument('--regressor_name', type=str, default='linear_regression', help='Regressor model to use.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON file containing regressor parameters.')
    parser.add_argument('--save_dir', type=str, default='./results/variance', help='The directory to save metrics JSON.')

    args = parser.parse_args()
    args_dict = vars(args)

    main(args.target_column, args.method, args.method_params, args.regressor_name, args.regressor_params, args.save_dir)