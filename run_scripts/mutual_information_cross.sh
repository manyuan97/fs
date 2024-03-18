python run_info_cross_validation.py --target_column y1  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_LR_y1 --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info_cross_validation.py --target_column y2  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_LR_y2 --regressor_name linear_regression --regressor_params configs/linear_regression.json
python run_info_cross_validation.py --target_column y3  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_LR_y3 --regressor_name linear_regression --regressor_params configs/linear_regression.json

python run_info_cross_validation.py --target_column y1  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_XG_y1 --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info_cross_validation.py --target_column y2  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_XG_y2 --regressor_name xgboost --regressor_params configs/xgboost.json
python run_info_cross_validation.py --target_column y3  --feature_selection mutual_info_regression --save_dir results/mutual_info_regression_cross_XG_y3 --regressor_name xgboost --regressor_params configs/xgboost.json

