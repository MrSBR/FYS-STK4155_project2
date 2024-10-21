import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import git
import sys
import autograd.numpy as anp
from autograd import grad
import jax.numpy as jnp
from jax import grad
path_to_root = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(path_to_root)

# Calculate OLSbeta and Ridgebeta. We are substituting the inversion of matrixes with the gradient descent method
def beta_OLS(X: np.ndarray, z: np.ndarray) -> np.ndarray:
	beta = np.linalg.pinv(X.T @ X) @ X.T @ z
	return beta

def beta_ridge(X: np.ndarray, z: np.ndarray, lamda: float) -> np.ndarray:
	I = np.eye(X.shape[1])
	betaRidge =  np.linalg.inv(X.T @ X + I*lamda) @ X.T @ z
	return betaRidge

# Calculating MSE and R2
def mse(true: np.ndarray, pred) -> np.ndarray:
	mse = np.mean((true - pred)**2)
	return mse

def r2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
	r2 = 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))
	return r2

def scale_train_test(train: np.ndarray, test:np.ndarray, with_std:bool=True, with_mean:bool=True) -> np.ndarray:
	scaler = StandardScaler(with_std=with_std, with_mean=with_mean) #Subtracting the mean and dividing by the standard deviation to scale/normalize the data
	train = scaler.fit_transform(train)
	test = scaler.transform(test)
	return train, test

def save_to_results(filename: str) -> None:
	plt.savefig(fname = path_to_root+'/results/'+filename)
	


