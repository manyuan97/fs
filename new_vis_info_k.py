def load_results(folder_path):
    results = {
        'train': [],
        'val': [],
        'test': [],
        'train_pos': [],
        'val_pos': [],
        'test_pos': []
    }
    ks = []
    selected_features = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith('thre_') and file_name.endswith('.json'):
            k = float(file_name.split('_')[1].split('.json')[0])
            
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            temp_results = {key: data['results'][key]['pearson_corr_without_mean'] for key in results}
            n_features = data['n_features']
            
            ks.append(k)
            for key in results:
                results[key].append(temp_results[key])
            selected_features.append(n_features)

    # 使用selected_features进行排序
    sorted_indices = sorted(range(len(selected_features)), key=lambda i: selected_features[i])
    ks = [ks[i] for i in sorted_indices]
    for key in results:
        results[key] = [results[key][i] for i in sorted_indices]
    selected_features = [selected_features[i] for i in sorted_indices]

    return ks, results, selected_features
