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
     

#Implementing plotting code
def plot_mse_and_r2(mse_array, r2_array, learning_rates, epochs_list):
    """
    Plots MSE and R^2 vs learning rate for each epoch.

    Parameters:
    - mse_array: 2D numpy array of MSE values (learning_rates x epochs)
    - r2_array: 2D numpy array of R^2 values (learning_rates x epochs)
    - learning_rates: list of learning rates used
    - epochs_list: list of epochs used
    """
    
    plt.figure(figsize=(12, 5))

    # Plot MSE
    plt.subplot(1, 2, 1)
    for idx, epoch in enumerate(epochs_list):
        plt.plot(learning_rates, mse_array[:, idx], marker='o', linestyle='-', label=f'Epochs = {epoch}')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('MSE')
        plt.title('MSE vs Learning Rate')
        plt.legend()
        plt.grid(True)

    # Plot R^2
    plt.subplot(1, 2, 2)
    for idx, epoch in enumerate(epochs_list):
        plt.plot(learning_rates, r2_array[:, idx], marker='o', linestyle='-', label=f'Epochs = {epoch}')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('R² Score')
        plt.title('R² vs Learning Rate')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_learning_rate(accuracy_array, learning_rates, epochs_list, title):
    """
    Plots accuracy vs. learning rate for all epochs in the same figure.
    
    Parameters:
    - accuracy_array: 2D numpy array of accuracy values (learning_rates x epochs)
    - learning_rates: list of learning rates used
    - epochs_list: list of epochs used
    """
    plt.figure(figsize=(10, 6))
    
    for idx, epoch in enumerate(epochs_list):
        plt.plot(learning_rates, accuracy_array[:, idx], marker='o', linestyle='-', label=f'Epochs = {epoch}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title(f'{title}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mse_and_r2_logisticreg(mse_array, r2_array, x_values, x_label, legend_values, legend_label, x_scale='log'):
    """
    Plots MSE and R^2 vs x_values for different legend_values.

    Parameters:
    - mse_array: 2D numpy array of MSE values (len(x_values) x len(legend_values))
    - r2_array: 2D numpy array of R^2 values (len(x_values) x len(legend_values))
    - x_values: list of x-axis values
    - x_label: label for the x-axis
    - legend_values: list of values to use in the legend
    - legend_label: label for the legend entries
    - x_scale: scale for x-axis ('linear' or 'log')
    """
    plt.figure(figsize=(12, 5))

    # Plot MSE
    plt.subplot(1, 2, 1)
    for idx, legend_val in enumerate(legend_values):
        plt.plot(x_values, mse_array[:, idx], marker='o', linestyle='-', label=f'{legend_label} = {legend_val}')
    if x_scale == 'log':
        plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('MSE')
    plt.title(f'MSE vs {x_label}')
    plt.legend()
    plt.grid(True)

    # Plot R^2
    plt.subplot(1, 2, 2)
    for idx, legend_val in enumerate(legend_values):
        plt.plot(x_values, r2_array[:, idx], marker='o', linestyle='-', label=f'{legend_label} = {legend_val}')
    if x_scale == 'log':
        plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('R² Score')
    plt.title(f'R² vs {x_label}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_learning_rates_logisticregression(accuracy_array, learning_rates, epochs_list, lambda_reg):
    """
    Plots accuracy vs learning rate for different epochs.

    Parameters:
    - accuracy_array: 2D numpy array of accuracy values (learning_rates x epochs_list)
    - learning_rates: list of learning rates used
    - epochs_list: list of epochs used
    """
    plt.figure(figsize=(8, 6))
    for idx, epoch in enumerate(epochs_list):
        plt.plot(learning_rates, accuracy_array[:, idx], marker='o', linestyle='-', label=f'Epochs = {epoch}'
        )

    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Learning Rate Logistic Regression (Lambda = {lambda_reg})')
    plt.legend()
    plt.grid(True)
    plt.show()




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


def cost_mse(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target) #changed to cost mse

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cost_cs(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target) #cost cs

def cost_bce(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return binary_cross_entropy(predict, target)

def binary_cross_entropy(predict, target):
    epsilon = 1e-15  # Small value to prevent log(0)
    # Clip predictions to avoid log(0) and division by zero
    predict = np.clip(predict, epsilon, 1 - epsilon)
    # Compute binary cross-entropy loss
    loss = - (target * np.log(predict) + (1 - target) * np.log(1 - predict))
    return np.mean(loss)

def train_network(inputs, targets, layers, activation_funcs, cost, learning_rate=0.001, epochs=1000):
    gradient_func = grad(cost, 1)  # Gradient wrt the second input (layers)
    
    for i in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        for idx, ((W, b), (W_g, b_g)) in enumerate(zip(layers, layers_grad)):
            W -= learning_rate * W_g
            b -= learning_rate * b_g
            layers[idx] = (W, b)  # Ensure that each layer is updated as a tuple (W, b)
        
        # Print layer shapes during training for debugging
        if i % 100 == 0:
            print(f"Epoch {i}: Layer shapes:")
            for idx, (W, b) in enumerate(layers):
                print(f"Layer {idx}: W shape = {W.shape}, b shape = {b.shape}")
    
    return layers


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for i, ((W, b), activation_func) in enumerate(zip(layers, activation_funcs)):
        print(f"Layer {i}: W shape = {W.shape}, b shape = {b.shape}")
        z = np.einsum("ij,kj->ki", W, a) + b
        a = activation_func(z)
    return a

