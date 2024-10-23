import numpy as np
import pandas as pd

""" File creating dataset so we use the same throughout the project"""

def load_simple_data(N_samples: int = 100, noise: float = 0.0, seed: int = 2024):
    np.random.seed(seed)
    x = np.random.randn(N_samples)
    a_true = np.random.rand(3)  # True coefficients a_0, a_1, a_2
    y = (a_true[0] + a_true[1] * x + a_true[2] * x**2 + np.random.randn(N_samples) * noise)  # y with some noise
    X = np.column_stack((np.ones(N_samples), x, x**2))
    return X, y, x, a_true

def load_wisconsin_data():
    # Prepare the dataset
    data = pd.read_csv('../../data/Wisconsin.csv')

    # Select features and target (radius_mean and texture_mean, diagnosis)
    X = data[['radius_mean', 'texture_mean']].values
    y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0).values.reshape(-1, 1)

    return X, y

