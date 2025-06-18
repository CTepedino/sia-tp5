import os
import copy
import numpy as np
from datetime import datetime

class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative=lambda x: 1,
                 optimizer="gradient"):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        
        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        # Inicialización de Xavier/Glorot para los pesos
        self.weights = []

        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1  # +1 por bias
            # Xavier/Glorot initialization: scale = sqrt(2.0 / (fan_in + fan_out))
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)).tolist())

        self.min_error = None
        self.best_weights = None
        self.patience = 8000
        self.patience_counter = 0

        # Parámetros para ADAM
        if self.optimizer == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = learning_rate  # Tasa de aprendizaje base
            self.m = [np.zeros_like(np.array(w)) for w in self.weights]  # Primer momento
            self.v = [np.zeros_like(np.array(w)) for w in self.weights]  # Segundo momento
            self.t = 0  # Paso de tiempo

            # Ajustar parámetros para problemas simples
            if len(layers) <= 3 and max(layers) <= 10:  # Problemas simples como XOR o par/impar
                self.beta1 = 0.8  # Menor momentum
                self.beta2 = 0.9  # Menor adaptación
                self.epsilon = 1e-6  # Mayor epsilon para evitar divisiones por cero
                self.alpha = learning_rate * 0.1  # Learning rate más conservador

        # Parámetros para Momentum
        elif self.optimizer == "momentum":
            self.momentum = 0.9  # Factor de momentum
            self.velocity = [np.zeros_like(np.array(w)) for w in self.weights]  # Velocidad inicial

    def update_weights_gradient(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        self.weights[l] = np.array(self.weights[l]) - self.learning_rate * weight_gradients

    def update_weights_momentum(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        # Actualizar velocidad con momentum
        self.velocity[l] = self.momentum * self.velocity[l] - self.learning_rate * weight_gradients
        # Actualizar pesos usando la velocidad
        self.weights[l] = np.array(self.weights[l]) + self.velocity[l]

    def update_weights_adam(self, l, delta, activation):
        self.t += 1
        weight_gradients = np.outer(delta, activation)

        # Actualizar momentos
        self.m[l] = self.beta1 * self.m[l] + (1 - self.beta1) * weight_gradients
        self.v[l] = self.beta2 * self.v[l] + (1 - self.beta2) * np.square(weight_gradients)

        # Corregir sesgo
        m_hat = self.m[l] / (1 - self.beta1 ** self.t)
        v_hat = self.v[l] / (1 - self.beta2 ** self.t)

        # Actualizar pesos con learning rate adaptativo
        update = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Limitar el tamaño de la actualización para problemas simples
        if len(self.layers) <= 3 and max(self.layers) <= 10:
            update = np.clip(update, -0.1, 0.1)  # Limitar actualizaciones grandes

        self.weights[l] = np.array(self.weights[l]) - update

    def forward_propagation(self, input_data):
        activations = [np.array(input_data)]
        hidden_states = []

        for layer_index in range(len(self.weights)):
            # Agregar bias como una columna
            prev_activation = np.append(activations[-1], 1.0)
            layer_weights = np.array(self.weights[layer_index])

            # Calcular salida de la capa
            h = np.dot(layer_weights, prev_activation)
            a = np.array([self.activator_function(x) for x in h])

            hidden_states.append(h)
            activations.append(a)

        return hidden_states, activations

    def back_propagation(self, expected_output, hidden_states, activations):
        deltas = [None] * len(self.weights)
        expected_output = np.array(expected_output)

        # Delta de la última capa
        last_layer = len(self.weights) - 1
        output_error = activations[-1] - expected_output
        deltas[last_layer] = output_error * np.array([self.activator_derivative(z) for z in hidden_states[-1]])

        # Backpropagation
        for l in range(len(self.weights) - 2, -1, -1):
            next_weights = np.array(self.weights[l + 1])
            next_delta = deltas[l + 1]
            current_h = hidden_states[l]

            # Calcular delta sin considerar el bias
            delta = np.dot(next_weights[:, :-1].T, next_delta) * np.array(
                [self.activator_derivative(z) for z in current_h])
            deltas[l] = delta

        # Actualización de pesos
        for l in range(len(self.weights)):
            delta = deltas[l]
            # Agregar bias como una columna
            activation = np.append(activations[l], 1.0)

            # Actualizar pesos según el optimizador seleccionado
            if self.optimizer == "adam":
                self.update_weights_adam(l, delta, activation)
            elif self.optimizer == "momentum":
                self.update_weights_momentum(l, delta, activation)
            else:  # gradient descent por defecto
                self.update_weights_gradient(l, delta, activation)

    def train(self, training_set, expected_outputs, epochs):
        best_error = float('inf')
        error_history = []
        min_delta = 1e-5
        window_size = 10

        # Inicializar ADAM si es necesario
        if self.optimizer == "adam":
            self.t = 0
            self.m = [np.zeros_like(np.array(w)) for w in self.weights]
            self.v = [np.zeros_like(np.array(w)) for w in self.weights]

        for epoch in range(epochs):
            error = 0
            np.random.seed(epoch)
            indices = np.random.permutation(len(training_set))

            for idx in indices:
                x = training_set[idx]
                y = expected_outputs[idx]

                hidden_states, activations = self.forward_propagation(x)
                output = activations[-1]
                self.back_propagation(y, hidden_states, activations)

                #error += 0.5 * np.sum((np.array(y) - output) ** 2)
                error += np.mean((np.array(y) - output) ** 2)

            average_error = error / len(training_set)
            error_history.append(average_error)

            if average_error < (best_error - min_delta):
                best_error = average_error
                self.best_weights = copy.deepcopy(self.weights)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

                if len(error_history) >= window_size:
                    recent_errors = error_history[-window_size:]
                    if all(recent_errors[i] > recent_errors[i - 1] * 1.01 for i in range(1, len(recent_errors))):
                        print(
                            f"Early stopping en época {epoch + 1} - Error aumenta consistentemente en {window_size} épocas")
                        break

                if self.patience_counter >= self.patience:
                    print(
                        f"Early stopping en época {epoch + 1} - Paciencia agotada después de {self.patience} épocas sin mejora")
                    break

        self.weights = copy.deepcopy(self.best_weights)

    def test(self, input_data):
        hidden_states, activations = self.forward_propagation(input_data)
        return activations[-1].tolist()
