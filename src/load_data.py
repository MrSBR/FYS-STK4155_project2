import numpy as np

""" File creating dataset so we use the same throughout the project"""

def load_simple_data(N_samples: int = 100, noise: float = 0.1, seed: int = 2024):
    np.random.seed(seed)
    X = np.random.rand(3, N_samples)  # Random matrix with 3 columns and N_samples rows
    a_true = np.random.rand(3)  # True coefficients a_0, a_1, a_2
    y = a_true[0] + a_true[1] * X + a_true[2] * X**2 + np.random.randn(N_samples) * 2  # y with some noise
    return X, y

def load_wisconsin_data():
    data = np.loadtxt("../data/wisconsin_data.csv", delimiter=",")
    X = data[:, 1:]
    y = data[:, 0]
    return X, y