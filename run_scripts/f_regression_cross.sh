python run_info.py --target_column y1  --feature_selection f_regression --save_dir results/f_regression_cross_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info.py --target_column y2  --feature_selection f_regression --save_dir results/f_regression_cross_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info.py --target_column y3  --feature_selection f_regression --save_dir results/f_regression_cross_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json

python run_info.py --target_column y1  --feature_selection f_regression --save_dir results/f_regression_cross_XG --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info.py --target_column y2  --feature_selection f_regression --save_dir results/f_regression_cross_XG --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info.py --target_column y3  --feature_selection f_regression --save_dir results/f_regression_cross_XG --regressor_name xgboost --regressor_params configs/xgboost.json

