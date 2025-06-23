import numpy as np

import vectorizedActivatorFunctions as activators

class Layer:
    def __init__(self, input_dim, output_dim, activator_function, activator_derivative, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activator_function = activator_function
        self.activator_derivative = activator_derivative
        self.learning_rate = learning_rate

        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weight = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.bias = np.zeros(output_dim)

        # Adam parameters
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

        self.m_w = np.zeros_like(self.weight)
        self.v_w = np.zeros_like(self.weight)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0

    def feedforward(self, input):
        if input.ndim == 1:
            input = input.reshape((input.shape[0], 1))

        self.input = input
        self.z = np.dot(self.weight, self.input) + np.tile(self.bias, (self.input.shape[1], 1)).T
        self.a = self.activator_function(self.z)
        return self.a

    def backward(self, last_gradient, output_layer=False):
        old_weight = np.copy(self.weight)
        if not output_layer:
            last_gradient *= self.activator_derivative(self.z)

        grad_weight = np.dot(last_gradient, self.input.T)
        grad_bias = np.sum(last_gradient, axis=1)

        self.t += 1

        self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * grad_weight
        self.v_w = self.beta_2 * self.v_w + (1 - self.beta_2) * (grad_weight ** 2)
        m_hat_w = self.m_w / (1 - self.beta_1 ** self.t)
        v_hat_w = self.v_w / (1 - self.beta_2 ** self.t)

        self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * grad_bias
        self.v_b = self.beta_2 * self.v_b + (1 - self.beta_2) * (grad_bias ** 2)
        m_hat_b = self.m_b / (1 - self.beta_1 ** self.t)
        v_hat_b = self.v_b / (1 - self.beta_2 ** self.t)

        self.weight -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        self.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        self.gradient = np.dot(old_weight.T, last_gradient)
        return self.gradient


class LatentLayer:
    def __init__(self, input_dim=1, output_dim=1, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean = Layer(
            self.input_dim, self.output_dim,
            activator_function=activators.identity,
            activator_derivative=activators.identity_derivative,
            learning_rate=learning_rate
        )
        self.log_var = Layer(
            self.input_dim, self.output_dim,
            activator_function=activators.identity,
            activator_derivative=activators.identity_derivative,
            learning_rate=learning_rate
        )

    def feedforward(self, input):
        self.mu = self.mean.feedforward(input)
        self.logvar = self.log_var.feedforward(input)
        self.epsilon = np.random.standard_normal(size=(self.output_dim, input.shape[1]))
        self.sample = self.mu + np.exp(self.logvar / 2.0) * self.epsilon

        return self.sample

    def backpropagate(self, last_gradient):
        normalizer = self.output_dim * last_gradient.shape[1]

        grad_log_var_kl = (np.exp(self.logvar) - 1) / (2 * normalizer)
        grad_mean_kl = self.mu / normalizer

        grad_log_var_mse = 0.5 * last_gradient * self.epsilon * np.exp(self.logvar / 2.0)
        grad_mean_mse = last_gradient

        grad_log_var_total = grad_log_var_kl + grad_log_var_mse
        grad_mean_total = grad_mean_kl + grad_mean_mse

        return self.mean.backward(grad_mean_total) + self.log_var.backward(grad_log_var_total)
