python run_unsupverised.py --target_column y1 --method pca --method_params configs/pca.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/pca_LR_y1
python run_unsupverised.py --target_column y2 --method pca --method_params configs/pca.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/pca_LR_y2
python run_unsupverised.py --target_column y3 --method pca --method_params configs/pca.json --regressor_name linear_regression --regressor_params configs/linear_regression.json --save_dir results/pca_LR_y3

python run_unsupverised.py --target_column y1 --method pca --method_params configs/pca.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/pca_xgboost_y1
python run_unsupverised.py --target_column y2 --method pca --method_params configs/pca.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/pca_xgboost_y2
python run_unsupverised.py --target_column y3 --method pca --method_params configs/pca.json --regressor_name xgboost --regressor_params configs/xgboost.json --save_dir results/pca_xgboost_y3

