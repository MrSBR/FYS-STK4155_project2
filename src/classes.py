import numpy as np
import autograd.numpy as anp
from autograd import grad
import jax.numpy as jnp
from jax import grad as jax_grad
import torch.nn as nn

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
        
        self.velocity = self.np_module.zeros_like(self.beta)

    def _compute_gradient_analytical(self, beta, Xj, yj):
        #yj and beta are (100,1) and (3,1) or (100,) and (3,) respectively
        y_pred = Xj @ beta
        gradient = 1/self.n * Xj.T @ (y_pred - yj)
        if self.cost_function == 'ridge':
            gradient[1:] += self.lambda_param * beta[1:] / self.n #div to be scaled
             #avoid regularization of intercept
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
        for _ in range(self.epochs):
            gradient = self.compute_gradient(self.beta, self.X, self.y)
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.beta += self.velocity

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
            if self.cost_function in ['ols', 'ridge', 'logistic']:
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
        
        self.velocity = self.np_module.zeros_like(self.beta)

    def _compute_gradient_analytical(self, beta, Xj, yj):
        if self.cost_function in ['ols', 'ridge']:
            y_pred = Xj @ beta
            gradient = (1 / self.batch_size) * Xj.T @ (y_pred - yj)
            if self.cost_function == 'ridge':
                gradient += (self.lambda_param / self.batch_size) * beta
        elif self.cost_function == 'logistic':
            z = Xj @ beta
            p = 1 / (1 + self.np_module.exp(-z))
            gradient = (1 / self.batch_size) * Xj.T @ (p - yj)
            if self.lambda_param > 0:
                gradient[1:] += (self.lambda_param / self.batch_size) * beta[1:]
        else:
            raise ValueError("Unsupported cost_function")
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
        for _ in range(self.epochs):
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            for batch_start in range(0, self.n, self.batch_size):
                batch_end = min(batch_start+self.batch_size, self.n) #takes the remaining datapoints even if they are not batch_size amount
                y_batch = y_shuff[batch_start:batch_end]
                X_batch = X_shuff[batch_start:batch_end,:]
                gradient = self.compute_gradient(self.beta, X_batch, y_batch)
                self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
                self.beta += self.velocity

    def _adagrad(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        self.velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            for batch_start in range(0, self.n, self.batch_size):
                batch_end = min(batch_start+self.batch_size, self.n) #takes the remaining datapoints even if they are not batch_size amount
                y_batch = y_shuff[batch_start:batch_end]
                X_batch = X_shuff[batch_start:batch_end,:]
                gradient = self.compute_gradient(self.beta, X_batch, y_batch)
                G += gradient ** 2
                self.velocity = self.momentum * self.velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
                self.beta += self.velocity

    def _rmsprop(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        self.velocity = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            for batch_start in range(0, self.n, self.batch_size):
                batch_end = min(batch_start+self.batch_size, self.n) #takes the remaining datapoints even if they are not batch_size amount
                y_batch = y_shuff[batch_start:batch_end]
                X_batch = X_shuff[batch_start:batch_end,:]
                gradient = self.compute_gradient(self.beta, X_batch, y_batch)
                G = self.decay_rate * G + (1 - self.decay_rate) * gradient ** 2
                self.velocity = self.momentum * self.velocity - self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient
                self.beta += self.velocity

    def _adam(self):
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        m = self.np_module.zeros_like(self.beta)
        v = self.np_module.zeros_like(self.beta)
        t = 0
        for _ in range(self.epochs):
            indices = np.random.permutation(self.n)
            X_shuff = self.X[indices]
            y_shuff = self.y[indices]
            for batch_start in range(0, self.n, self.batch_size):
                t += 1
                batch_end = min(batch_start+self.batch_size, self.n) #takes the remaining datapoints even if they are not batch_size amount
                y_batch = y_shuff[batch_start:batch_end]
                X_batch = X_shuff[batch_start:batch_end,:]
                gradient = self.compute_gradient(self.beta, X_batch, y_batch)
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * gradient ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat

# Example usage:
# sgd = StochasticGradientDescent(X, y, beta, learning_rate=0.01, epochs=100, optimizer='adam',gradient_method='jax', lambda_param=0.1, cost_function='ridge')
# optimized_beta = sgd.optimize()


class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_reg=0.0, batch_size=None, optimizer='sgd', gradient_method='analytical', decay_rate=0.9):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.gradient_method = gradient_method
        self.decay_rate = decay_rate
        self.beta = None

    def fit(self, X, y):
        N, m = X.shape
        # Add bias term
        X_bias = np.hstack([np.ones((N, 1)), X])
        m = X_bias.shape[1]
        self.beta = np.zeros((m, 1))

        # Initialize the optimizer
        sgd = StochasticGradientDescent(
            X=X_bias,
            y=y,
            beta=self.beta,
            learning_rate=self.learning_rate,
            epochs=self.n_iter,
            momentum=0,
            optimizer=self.optimizer,
            gradient_method=self.gradient_method,
            lambda_param=self.lambda_reg,
            cost_function='logistic',
            batch_size=self.batch_size,
            decay_rate=self.decay_rate
        )

        # Optimize beta
        self.beta = sgd.optimize()
        return self

    def predict_proba(self, X):
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        z = X_bias @ self.beta
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# Define the neural network model
class RegClasNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation_function, last_layer_activation):
        super(RegClasNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.activation = activation_function
        self.lastlayeractivation = last_layer_activation

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)  # No activation on the output for regression (linear)
        out = self.lastlayeractivation(out)
        return out