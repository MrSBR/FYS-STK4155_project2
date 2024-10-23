import numpy as np
import pandas as pd

""" File creating dataset so we use the same throughout the project"""

def load_simple_data(N_samples: int = 100, noise: float = 0.0, seed: int = 2024):
    np.random.seed(seed)
    x = np.random.randn(N_samples)
    X = np.column_stack((np.ones(N_samples), x, x**2))
    a_true = np.random.rand(3)  # True coefficients a_0, a_1, a_2
    y = a_true[0] + a_true[1] * x + a_true[2] * x**2 + np.random.randn(N_samples) * noise  # y with some noise
    return X, y, x, a_true

def load_wisconsin_data():
    # Read the data into a pandas DataFrame
    data = pd.read_csv("../../data/Wisconsin.csv")

    # Change 'B' to 1 and 'M' to 2 in column 2
    data.loc[data['diagnosis'] == 'B', 'diagnosis'] = 1
    data.loc[data['diagnosis'] == 'M', 'diagnosis'] = 2

    x = data['area_mean']

    X = np.column_stack((np.ones(len(x)), x, x**2))
    y = data['diagnosis']
    return X, y

