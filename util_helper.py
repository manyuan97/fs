import math
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import json


class Utils:
    @staticmethod
    def timeit(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")
            return result

        return wrapper


def visualize_selected_features(selected_features, num_features, filename):
    dim = math.ceil(np.sqrt(num_features))

    image = np.zeros((dim ** 2,))

    image[:len(selected_features)] = selected_features.astype(int)

    image = image.reshape((dim, dim))

    # Plot and save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys', interpolation='nearest')
    plt.title('Selected Features Visualization')
    plt.axis('off')  # Hide the axes
    plt.savefig(filename)
    plt.close()

def load_params_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
