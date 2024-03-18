import json
import numpy as np

class ResultSaver:
    def __init__(self):
        self.encoder = self.NumpyEncode

    class NumpyEncode(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return json.JSONEncoder.default(self, obj)

    def save_metrics_as_json(self, metrics, file_path):
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4, cls=self.encoder)