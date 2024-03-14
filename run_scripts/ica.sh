python run_unsupverised.py --target_column y1 --method ica --method_params configs/ica.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/ica_y1_LR
python run_unsupverised.py --target_column y2 --method ica --method_params configs/ica.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/ica_y1_LR
python run_unsupverised.py --target_column y3 --method ica --method_params configs/ica.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/ica_y1_LR

python run_unsupverised.py --target_column y1 --method ica --method_params configs/ica.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/ica_y1_XG
python run_unsupverised.py --target_column y2 --method ica --method_params configs/ica.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/ica_y2_XG
python run_unsupverised.py --target_column y3 --method ica --method_params configs/ica.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/ica_y3_XG

