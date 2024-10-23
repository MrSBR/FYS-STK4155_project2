import numpy as np
import autograd.numpy as anp
from autograd import grad
import jax.numpy as jnp
from jax import grad as jax_grad
import torch
import torch.nn as nn
import torch.optim as optim

class GradientDescent:
    def __init__(self, X, y, beta, learning_rate=0.01, epochs=100, momentum=0,
                 optimizer='gd', gradient_method='analytical', lambda_param=0.0,
                 cost_function='ols', decay_rate=0.9):
        self.X = X
        self.y = y
        self.beta = beta.copy()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.optimizer = optimizer
        self.gradient_method = gradient_method
        self.lambda_param = lambda_param
        self.cost_function = cost_function
        self.n = len(y)
        self.decay_rate = decay_rate

        # Select the numpy module and gradient computation function
        if self.gradient_method == 'analytical':
            self.np_module = np
            self.compute_gradient = self._compute_gradient_analytical
        elif self.gradient_method == 'autograd':
            self.np_module = anp
            self.X = anp.array(self.X)
            self.y = anp.array(self.y)
            self.beta = anp.array(self.beta)
            self.compute_gradient = self._compute_gradient_autograd
        elif self.gradient_method == 'jax':
            self.np_module = jnp
            self.X = jnp.array(self.X)
            self.y = jnp.array(self.y)
            self.beta = jnp.array(self.beta)
            self.compute_gradient = self._compute_gradient_jax
        else:
            raise ValueError(f"Unknown gradient method: {self.gradient_method}")

    def _compute_gradient_analytical(self, beta, Xj, yj):
        y_pred = Xj @ beta
        gradient = 1/self.n * Xj.T @ (y_pred - yj)
        if self.cost_function == 'ridge':
            gradient += self.lambda_param * beta
        return gradient

    def _compute_loss_autograd(self, beta, Xj, yj):
        y_pred = anp.dot(Xj, beta)
        loss = 0.5 * anp.sum((y_pred - yj) ** 2)
        if self.cost_function == 'ridge':
            ridge_penalty = 0.5 * self.lambda_param * anp.sum(beta ** 2)
            loss += ridge_penalty
        return loss

    def _compute_gradient_autograd(self, beta, Xj, yj):
        compute_loss_local = lambda beta: self._compute_loss_autograd(beta, Xj, yj)
        return grad(compute_loss_local)(beta)

    def _compute_loss_jax(self, beta, Xj, yj):
        y_pred = jnp.dot(Xj, beta)
        loss = 0.5 * jnp.sum((y_pred - yj) ** 2)
        if self.cost_function == 'ridge':
            ridge_penalty = 0.5 * self.lambda_param * jnp.sum(beta ** 2)
            loss += ridge_penalty
        return loss

    def _compute_gradient_jax(self, beta, Xj, yj):
        compute_loss_local = lambda beta: self._compute_loss_jax(beta, Xj, yj)
        return jax_grad(compute_loss_local)(beta)

    def optimize(self):
        if self.optimizer == 'gd':
            self._gd()
        elif self.optimizer == 'adagrad':
            self._adagrad()
        elif self.optimizer == 'rmsprop':
            self._rmsprop()
        elif self.optimizer == 'adam':
            self._adam()
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        return self.beta

    def _gd(self):
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            velocity = self.momentum * velocity - self.learning_rate * gradient
            self.beta += velocity

    def _adagrad(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            G += gradient ** 2
            velocity = self.momentum * velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
            self.beta += velocity

    def _rmsprop(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            G = self.decay_rate * G + (1 - self.decay_rate) * gradient ** 2
            velocity = self.momentum * velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
            self.beta += velocity


    def _adam(self):
        #adam optimizer essentially includes momentum through first moment (m)
        #so it shouldn't be implemented alongside adam.
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        m = self.np_module.zeros_like(self.beta)
        v = self.np_module.zeros_like(self.beta)
        t = 0
        for _ in range(self.epochs):
            t += 1
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat

# Example usage:
# gd = GradientDescent(X, y, beta, learning_rate=0.01, epochs=100, optimizer='adam', gradient_method='jax', lambda_param=0.1, cost_function='ridge')
# optimized_beta = gd.optimize()


class StochasticGradientDescent:
    def __init__(self, X, y, beta, learning_rate=0.01, epochs=100, momentum=0, 
                 optimizer='sgd', gradient_method='analytical', lambda_param=0.0,
                 cost_function='ols', batch_size = None, decay_rate = 0.9):
        self.X = X
        self.y = y
        self.beta = beta.copy()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.optimizer = optimizer
        self.gradient_method = gradient_method
        self.lambda_param = lambda_param
        self.cost_function = cost_function
        self.n = len(y)
        self.decay_rate = decay_rate
        if batch_size == None:
            self.batch_size = len(y)//10
        else:
            self.batch_size = batch_size

        # Select the numpy module and gradient computation function
        if self.gradient_method == 'analytical':
            self.np_module = np
            self.compute_gradient = self._compute_gradient_analytical
        elif self.gradient_method == 'autograd':
            self.np_module = anp
            self.X = anp.array(self.X)
            self.y = anp.array(self.y)
            self.beta = anp.array(self.beta)
            self.compute_gradient = self._compute_gradient_autograd
        elif self.gradient_method == 'jax':
            self.np_module = jnp
            self.X = jnp.array(self.X)
            self.y = jnp.array(self.y)
            self.beta = jnp.array(self.beta)
            self.compute_gradient = self._compute_gradient_jax
        else:
            raise ValueError(f"Unknown gradient method: {self.gradient_method}")

    def _compute_gradient_analytical(self, beta, Xj, yj):
        y_pred = Xj @ beta
        gradient = 1/self.batch_size * Xj.T @ (y_pred - yj)
        if self.cost_function == 'ridge':
            gradient += self.lambda_param * beta
        return gradient

    def _compute_loss_autograd(self, beta, Xj, yj):
        y_pred = anp.dot(Xj, beta)
        loss = 0.5 * anp.sum((y_pred - yj) ** 2)
        if self.cost_function == 'ridge':
            ridge_penalty = 0.5 * self.lambda_param * anp.sum(beta ** 2)
            loss += ridge_penalty
        return loss

    def _compute_gradient_autograd(self, beta, Xj, yj):
        compute_loss_local = lambda beta: self._compute_loss_autograd(beta, Xj, yj)
        return grad(compute_loss_local)(beta)

    def _compute_loss_jax(self, beta, Xj, yj):
        y_pred = jnp.dot(Xj, beta)
        loss = 0.5 * jnp.sum((y_pred - yj) ** 2)
        if self.cost_function == 'ridge':
            ridge_penalty = 0.5 * self.lambda_param * jnp.sum(beta ** 2)
            loss += ridge_penalty
        return loss

    def _compute_gradient_jax(self, beta, Xj, yj):
        compute_loss_local = lambda beta: self._compute_loss_jax(beta, Xj, yj)
        return jax_grad(compute_loss_local)(beta)

    def optimize(self):
        if self.optimizer == 'sgd':
            self._sgd()
        elif self.optimizer == 'adagrad':
            self._adagrad()
        elif self.optimizer == 'rmsprop':
            self._rmsprop()
        elif self.optimizer == 'adam':
            self._adam()
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        return self.beta

    def _sgd(self):
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            #shuffle X and Y while maintaining pairs
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            batch_start = np.random.randint(0, self.n - self.batch_size)
            y_batch = y_shuff[batch_start:batch_start + self.batch_size]
            X_batch = X_shuff[batch_start:batch_start + self.batch_size,:]
            gradient = self.compute_gradient(self.beta, X_batch, y_batch)
            velocity = self.momentum * velocity - self.learning_rate * gradient
            self.beta += velocity

    def _adagrad(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            #shuffle X and Y while maintaining pairs
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            batch_start = np.random.randint(0, self.n - self.batch_size)
            y_batch = y_shuff[batch_start:batch_start + self.batch_size]
            X_batch = X_shuff[batch_start:batch_start + self.batch_size,:]
            gradient = self.compute_gradient(self.beta, X_batch, y_batch)
            G += gradient ** 2
            velocity = self.momentum * velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
            self.beta += velocity

    def _rmsprop(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            #shuffle X and Y while maintaining pairs
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            batch_start = np.random.randint(0, self.n - self.batch_size)
            y_batch = y_shuff[batch_start:batch_start + self.batch_size]
            X_batch = X_shuff[batch_start:batch_start + self.batch_size,:]
            gradient = self.compute_gradient(self.beta, X_batch, y_batch)
            G = self.decay_rate * G + (1 - self.decay_rate) * gradient ** 2
            velocity = self.momentum * velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
            self.beta += velocity

    def _adam(self):
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        m = self.np_module.zeros_like(self.beta)
        v = self.np_module.zeros_like(self.beta)
        t = 0
        for _ in range(self.epochs):
            t += 1
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            batch_start = np.random.randint(0, self.n - self.batch_size)
            y_batch = y_shuff[batch_start:batch_start + self.batch_size]
            X_batch = X_shuff[batch_start:batch_start + self.batch_size,:]
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat

# Example usage:
# sgd = StochasticGradientDescent(X, y, beta, learning_rate=0.01, epochs=100, optimizer='adam',gradient_method='jax', lambda_param=0.1, cost_function='ridge')
# optimized_beta = sgd.optimize()


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.beta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y):
        N = len(y)
        p = self.sigmoid(X @ self.beta)
        cost = (-1 / N) * (y.T @ np.log(p) + (1 - y).T @ np.log(1 - p))
        reg_term = (self.lambda_reg / (2 * N)) * np.sum(self.beta[1:] ** 2)
        return cost + reg_term

    def fit(self, X, y):
        N, m = X.shape
        self.beta = np.zeros(m).reshape(-1, 1)
        cost_history = []

        for _ in range(self.n_iter):
            p = self.sigmoid(X @ self.beta)
            gradient = (1 / N) * X.T @ (p - y)
            # Apply regularization (exclude bias term)
            gradient[1:] += (self.lambda_reg / N) * self.beta[1:]
            self.beta -= self.learning_rate * gradient
            cost = self.cost_function(X, y)
            cost_history.append(cost)

        return cost_history

    def predict(self, X):
        return self.sigmoid(X @ self.beta)


# Define the neural network model
class RegClasNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RegClasNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)  # No activation on the output for regression (linear)
        out = self.sigmoid(out)
        return out