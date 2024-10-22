#import numpy as np?
import autograd.numpy as np
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
from sklearn.metrics import accuracy_score

# Calculate OLSbeta and Ridgebeta. We are substituting the inversion of matrixes with the gradient descent method
def beta_OLS(X: np.ndarray, z: np.ndarray) -> np.ndarray:
	beta = np.linalg.pinv(X.T @ X) @ X.T @ z
	return beta

def beta_Ridge(X: np.ndarray, z: np.ndarray, lamda: float) -> np.ndarray:
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
	

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha*z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def linear(z):
     return z


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def accuracy(predictions, targets):
    # Convert predictions to label indices
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Check if targets are one-hot encoded or label indices
    if targets.ndim > 1 and targets.shape[1] > 1:
        # If targets are one-hot encoded, convert to label indices
        true_labels = np.argmax(targets, axis=1)
    else:
        # If targets are already label indices
        true_labels = targets.flatten()
    
    return accuracy_score(true_labels, predicted_labels)



def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

from autograd import grad
import autograd.numpy as anp

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cost_mse(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target) #changed to cost mse

def cost_cs(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target) #cost cs

def train_network(
    inputs, targets, layers, activation_funcs, cost, learning_rate=0.001, epochs=1000,
    ):
    gradient_func = grad(cost, 1)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers
    layers_grad = gradient_func(inputs, layers, activation_funcs, targets)  # Don't change this
    for i in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        for (W, b), (W_g, b_g) in zip(layers, layers_grad):
            W -= learning_rate * W_g
            b -= learning_rate * b_g
    return layers

def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = np.einsum("ij,kj->ki", W, a) + b
        a = activation_func(z)
    return a