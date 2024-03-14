python run_info.py --target_column y1 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info.py --target_column y2 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info.py --target_column y3 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_LR --regressor_name linear_regression --regressor_params configs/linear_regression.json

python run_info.py --target_column y1 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_XG --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info.py --target_column y2 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_XG --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info.py --target_column y3 --k 100 --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_y1_XG --regressor_name xgboost --regressor_params configs/xgboost.json

