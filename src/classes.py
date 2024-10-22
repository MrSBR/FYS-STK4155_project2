import numpy as np
import autograd.numpy as anp
from autograd import grad
import jax.numpy as jnp
from jax import grad as jax_grad

class GradientDescent:
    def __init__(self, X, y, beta, learning_rate=0.01, epochs=100, momentum=0,
                 optimizer='gd', gradient_method='analytical', lambda_param=0.0, cost_function='ols'):
        self.X = X
        self.y = y
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.optimizer = optimizer
        self.gradient_method = gradient_method
        self.lambda_param = lambda_param
        self.cost_function = cost_function
        self.n = len(y)
        self.prev_beta = beta.copy() if momentum == 1 else None

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
        gradient = Xj.T @ (y_pred - yj)
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
        for _ in range(self.epochs):
            for j in range(self.n):
                gradient = self.compute_gradient(self.beta, self.X[j], self.y[j])
                if self.momentum == 1:
                    self.beta -= self.learning_rate * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate * gradient

    def _adagrad(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            for j in range(self.n):
                gradient = self.compute_gradient(self.beta, self.X[j], self.y[j])
                G += gradient ** 2
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient

    def _rmsprop(self):
        epsilon = 1e-8
        decay_rate = 0.9
        G = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            for j in range(self.n):
                gradient = self.compute_gradient(self.beta, self.X[j], self.y[j])
                G = decay_rate * G + (1 - decay_rate) * gradient ** 2
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient

    def _adam(self):
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        m = self.np_module.zeros_like(self.beta)
        v = self.np_module.zeros_like(self.beta)
        t = 0
        for _ in range(self.epochs):
            for j in range(self.n):
                t += 1
                gradient = self.compute_gradient(self.beta, self.X[j], self.y[j])
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * gradient ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat

# Example usage:
# gd = GradientDescent(X, y, beta, learning_rate=0.01, epochs=100, optimizer='adam', gradient_method='jax', lambda_param=0.1, cost_function='ridge')
# optimized_beta = gd.optimize()


class StochasticGradientDescent:
    def __init__(self, X, y, beta, learning_rate=0.01, epochs=100, momentum=0, 
                 optimizer='sgd', gradient_method='analytical', lambda_param=0.0, cost_function='ols'):
        self.X = X
        self.y = y
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.optimizer = optimizer
        self.gradient_method = gradient_method
        self.lambda_param = lambda_param
        self.cost_function = cost_function
        self.n = len(y)
        self.prev_beta = beta.copy() if momentum == 1 else None

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
        gradient = Xj.T @ (y_pred - yj)
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
        for _ in range(self.epochs):
            for _ in range(self.n):
                random_index = np.random.randint(self.n)
                X_random = self.X[random_index].reshape(1, -1)
                y_random = self.y[random_index]
                gradient = self.compute_gradient(self.beta, X_random, y_random)
                if self.momentum == 1:
                    self.beta -= self.learning_rate * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate * gradient

    def _adagrad(self):
        epsilon = 1e-8
        G = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            for _ in range(self.n):
                random_index = np.random.randint(self.n)
                X_random = self.X[random_index].reshape(1, -1)
                y_random = self.y[random_index]
                gradient = self.compute_gradient(self.beta, X_random, y_random)
                G += gradient ** 2
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient

    def _rmsprop(self):
        epsilon = 1e-8
        decay_rate = 0.9
        G = self.np_module.zeros_like(self.beta)
        for _ in range(self.epochs):
            for _ in range(self.n):
                random_index = np.random.randint(self.n)
                X_random = self.X[random_index].reshape(1, -1)
                y_random = self.y[random_index]
                gradient = self.compute_gradient(self.beta, X_random, y_random)
                G = decay_rate * G + (1 - decay_rate) * gradient ** 2
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(G) + epsilon) * gradient

    def _adam(self):
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        m = self.np_module.zeros_like(self.beta)
        v = self.np_module.zeros_like(self.beta)
        t = 0
        for _ in range(self.epochs):
            for _ in range(self.n):
                t += 1
                random_index = np.random.randint(self.n)
                X_random = self.X[random_index].reshape(1, -1)
                y_random = self.y[random_index]
                gradient = self.compute_gradient(self.beta, X_random, y_random)
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * gradient ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                if self.momentum == 1:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat + self.momentum * (self.beta - self.prev_beta)
                    self.prev_beta = self.beta.copy()
                else:
                    self.beta -= self.learning_rate / (self.np_module.sqrt(v_hat) + epsilon) * m_hat

# Example usage:
# sgd = StochasticGradientDescent(X, y, beta, learning_rate=0.01, epochs=100, optimizer='adam',gradient_method='jax', lambda_param=0.1, cost_function='ridge')
# optimized_beta = sgd.optimize()

