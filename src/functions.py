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

def mse_derivative(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (predictions - targets) / targets.shape[0]

def r2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
	r2 = 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))
	return r2

#scale the train and test data
def scale_train_test(train: np.ndarray, test:np.ndarray, with_std:bool=True, with_mean:bool=True) -> np.ndarray:
	scaler = StandardScaler(with_std=with_std, with_mean=with_mean) #Subtracting the mean and dividing by the standard deviation to scale/normalize the data
	train = scaler.fit_transform(train)
	test = scaler.transform(test)
	return train, test

#save plot to results
def save_to_results(filename: str) -> None:
	plt.savefig(fname = path_to_root+'/results/'+filename)
     

#Implementing plotting code
def plot_mse_and_r2(mse_array, r2_array, learning_rates, epochs_list, filename):
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
    plt.savefig(f'../results/{filename}.png')
    plt.show()

def plot_accuracy_vs_learning_rate(accuracy_array, learning_rates, epochs_list, title, filename):
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
    plt.savefig(f'../results/{filename}.png')
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


def plot_accuracy_vs_learning_rates_logisticregression(accuracy_array, learning_rates, epochs_list, lambda_reg, filename):
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
    plt.savefig(f'../results/{filename}.png')
    plt.show()




# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha*z)

def leaky_ReLU_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def linear(z):
     return z

def linear_derivative(z):
    return np.ones_like(z)

def softmax(z):
    """Compute softmax values for each set of scores in z."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_derivative(sigma, delta_next):
    """
    Compute the derivative of the softmax function.
    
    Parameters:
    - sigma: The output of the softmax function, shape (batch_size, num_classes)
    - delta_next: The gradient of the loss with respect to the output of softmax, shape (batch_size, num_classes)
    
    Returns:
    - delta_current: The gradient of the loss with respect to the input of softmax, shape (batch_size, num_classes)
    """
    # Compute the dot product of delta_next and sigma along classes axis
    dot_product = np.sum(delta_next * sigma, axis=1, keepdims=True)  # Shape: (batch_size, 1)
    
    # Compute delta_current
    delta_current = sigma * (delta_next - dot_product)
    return delta_current


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


    #Cross Entropy
def ce(predictions, targets):
    epsilon = 1e-15  # Small value to prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / targets.shape[0]
    return loss

    #Cross Entropy derivative
def ce_derivative(predictions, targets):
    epsilon = 1e-15  # Prevent division by zero
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    derivative = - (targets / predictions)
    return derivative / targets.shape[0]

    #Binary cross entropy
def bce(predictions, targets):
    epsilon = 1e-15  # Small value to prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = - (targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return np.mean(loss)

    #Binary cross entropy derivative
def bce_derivative(predictions, targets):
    epsilon = 1e-15  # Prevent division by zero
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    derivative = - (targets / predictions) + (1 - targets) / (1 - predictions)
    derivative /= targets.shape[0]  # Average over the batch
    return derivative

    #create a batch of layers for a neural network with random weights and biases
def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

"""def train_network(inputs, targets, layers, activation_funcs, cost, learning_rate=0.001, epochs=1000):
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
    
    return layers"""

    #trains network
def train_network(inputs, targets, layers, activation_funcs, activation_ders, cost, cost_der, learning_rate=0.001, epochs=1000):
    for epoch in range(epochs):
        # Forward pass
        predictions = feed_forward_batch(inputs, layers, activation_funcs)
        
        # Backward pass (compute gradients)
        layer_grads = backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der)
        
        # Update weights and biases
        for idx, ((W, b), (dW, db)) in enumerate(zip(layers, layer_grads)):
            W -= learning_rate * dW
            b -= learning_rate * db
            layers[idx] = (W, b)
        
        # Optionally print loss and layer shapes for debugging
        if epoch % 100 == 0:
            loss = cost(predictions, targets)
            print(f"Epoch {epoch}, Loss: {loss}")
            for idx, (W, b) in enumerate(layers):
                print(f"Layer {idx}: W shape = {W.shape}, b shape = {b.shape}")
    
    return layers

    #feeds a batch of inputs forward
def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for i, ((W, b), activation_func) in enumerate(zip(layers, activation_funcs)):
        print(f"Layer {i}: W shape = {W.shape}, b shape = {b.shape}")
        z = np.einsum("ij,kj->ki", W, a) + b
        a = activation_func(z)
    return a


"""def backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der):
    # Forward pass with saving intermediate values
    layer_inputs, zs, predictions = feed_forward_saver(inputs, layers, activation_funcs)
    
    # Initialize gradients list
    layer_grads = [None] * len(layers)
    
    # Compute error at output layer
    delta = cost_der(predictions, targets) * activation_ders[-1](zs[-1])
    
    # Compute gradients for output layer
    dW = np.dot(delta.T, layer_inputs[-1])
    db = np.sum(delta, axis=0)
    layer_grads[-1] = (dW, db)
    
    # Backpropagate the error
    for i in reversed(range(len(layers) - 1)):
        W_next = layers[i + 1][0]
        delta = np.dot(delta, W_next) * activation_ders[i](zs[i])
        dW = np.dot(delta.T, layer_inputs[i])
        db = np.sum(delta, axis=0)
        layer_grads[i] = (dW, db)
    
    return layer_grads"""

    #performs backpropagation
def backpropagation(inputs, layers, activation_funcs, targets, activation_ders, cost_der=None):
    # Forward pass with saving intermediate values
    layer_inputs, zs, predictions = feed_forward_saver(inputs, layers, activation_funcs)
    
    # Initialize gradients list
    layer_grads = [None] * len(layers)
    
    # Compute error at output layer
    if activation_funcs[-1] == softmax:
        # Compute the gradient of the loss with respect to the softmax output
        delta_next = cost_der(predictions, targets)
        # Compute the gradient with respect to z (input of softmax)
        delta = softmax_derivative(predictions, delta_next)
    else:
        delta_next = cost_der(predictions, targets)
        delta = delta_next * activation_ders[-1](zs[-1])
    
    # Compute gradients for output layer
    dC_dW = np.dot(delta.T, layer_inputs[-1])  # Shape: (output_size, hidden_size)
    dC_db = np.sum(delta, axis=0)
    layer_grads[-1] = (dC_dW, dC_db)
    
    # Backpropagate the error through previous layers
    for i in reversed(range(len(layers) - 1)):
        W_next = layers[i + 1][0]
        delta = np.dot(delta, W_next) * activation_ders[i](zs[i])
        dC_dW = np.dot(delta.T, layer_inputs[i])
        dC_db = np.sum(delta, axis=0)
        layer_grads[i] = (dC_dW, dC_db)
    
    return layer_grads

    #performs forward pass and additionally returns the input of each layer
def feed_forward_saver(inputs, layers, activation_funcs):
    a = inputs
    layer_inputs = [a]
    zs = []
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = np.dot(a, W.T) + b
        zs.append(z)
        a = activation_func(z)
        layer_inputs.append(a)
    return layer_inputs[:-1], zs, a
