import argparse
import gc
import json
import os

from data_helper import DataProcessor
from eval_helper import ModelEvaluator
from model_helper import FeatureSelectorByModel
from model_helper import ModelTrainer
from result_helper import ResultSaver
from util_helper import visualize_selected_features,load_params_from_file


def main(target_column, model_name, model_params_path, regressor_name, regressor_params_path, threshold, save_dir):
    os.makedirs(save_dir, exist_ok=True)


    model_params = load_params_from_file(model_params_path)
    regressor_params = load_params_from_file(regressor_params_path)


    # 定义文件路径
    train_file_path = './train.parquet'
    val_file_path = './val.parquet'
    test_file_path = './test.parquet'

    # 初始化数据处理器
    data_processor = DataProcessor(target_column=target_column)

    # 加载和预处理数据
    X_train, y_train, scaler = data_processor.load_and_preprocess_data(train_file_path)

    model_trainer = ModelTrainer(model_name=model_name, params=model_params)

    # 训练模型
    xgboost_model = model_trainer.train_model(X_train, y_train)

    # 选择特征
    feature_selector = FeatureSelectorByModel(xgboost_model)
    selector, X_train_selected, selected_features = feature_selector.select_features(X_train, threshold=threshold,
                                                                                     prefit=True)
    visualize_path = os.path.join(save_dir, f'selected_features.png')
    visualize_selected_features(selected_features, X_train.shape[1], visualize_path)

    new_model_trainer = ModelTrainer(model_name=regressor_name, params=regressor_params)
    regressor = new_model_trainer.train_model(X_train_selected, y_train)

    # 评估模型
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

    metrics = {
        'model_name': model_name,
        'results': {
            'train': metrics_train.__dict__,
            'val': metrics_val.__dict__,
            'test': metrics_test.__dict__},
        'args': args_dict

    }

    # 保存结果
    result_saver = ResultSaver()
    json_file_path = os.path.join(save_dir, 'results.json')
    result_saver.save_metrics_as_json(metrics, json_file_path)

    print(f"Metrics stored in {json_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models with feature selection.')
    parser.add_argument('--target_column', type=str, default='y1', help='The target column for prediction.')
    parser.add_argument('--model_name', type=str, default='xgboost', help='The target column for prediction.')
    parser.add_argument('--regressor_name', type=str, default='linear_regression', help='The target column for prediction.')
    parser.add_argument('--regressor_params', type=str, default='./configs/linear_regression.json',
                        help='File path to JSON string of regressor parameters.')
    parser.add_argument('--model_params', type=str, default='./configs/xgboost_config.json',
                        help='File path to JSON string of model parameters.')
    parser.add_argument('--threshold', type=str, default='1.25*mean', help='Threshold for feature selection.')
    parser.add_argument('--save_dir', type=str, default='./results/xgboost',
                        help='The file path to save metrics JSON.')

    args = parser.parse_args()
    args_dict = vars(args)

    main(args.target_column, args.model_name, args.model_params, args.regressor_name, args.regressor_params,
         args.threshold, args.save_dir)
