import os
import copy
import numpy as np
from datetime import datetime

class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative=lambda x: 1,
                 optimizer="gradient", adaptive_lr=True, loss_function="mse"):
        self.layers = layers
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Guardar el learning rate inicial
        self.optimizer = optimizer
        self.loss_function = loss_function

        
        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        # Parámetros para learning rate adaptativo
        self.adaptive_lr = adaptive_lr
        self.lr_patience = 500  # Épocas sin mejora antes de reducir LR
        self.lr_factor = 0.5  # Factor por el cual reducir el LR
        self.lr_min = 1e-6  # Learning rate mínimo
        self.lr_max = learning_rate * 10  # Learning rate máximo
        self.lr_improvement_threshold = 1e-6  # Umbral para considerar mejora
        self.lr_boost_factor = 1.1  # Factor para aumentar LR cuando hay mejoras
        self.lr_boost_patience = 400  # Épocas con mejora antes de aumentar LR
        self.lr_boost_counter = 0
        self.lr_reduce_counter = 0

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
        current_lr = self.learning_rate if self.optimizer != "adam" else self.alpha
        update = current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Limitar el tamaño de la actualización para problemas simples
        if len(self.layers) <= 3 and max(self.layers) <= 10:
            update = np.clip(update, -0.1, 0.1)  # Limitar actualizaciones grandes

        self.weights[l] = np.array(self.weights[l]) - update

    def adjust_learning_rate(self, current_error, best_error):
        """Ajusta el learning rate basándose en el progreso del entrenamiento"""
        if not self.adaptive_lr:
            return

        improvement = best_error - current_error
        
        if improvement > self.lr_improvement_threshold:
            # Hay mejora, resetear contadores y posiblemente aumentar LR
            self.lr_reduce_counter = 0
            self.lr_boost_counter += 1
            
            if self.lr_boost_counter >= self.lr_boost_patience:
                # Aumentar learning rate
                old_lr = self.learning_rate
                self.learning_rate = min(self.learning_rate * self.lr_boost_factor, self.lr_max)
                if self.optimizer == "adam":
                    self.alpha = min(self.alpha * self.lr_boost_factor, self.lr_max)
                
                if self.learning_rate != old_lr:
                    print(f"  ↑ Learning rate aumentado a: {self.learning_rate}")
                
                self.lr_boost_counter = 0
        else:
            # No hay mejora significativa
            self.lr_boost_counter = 0
            self.lr_reduce_counter += 1
            
            if self.lr_reduce_counter >= self.lr_patience:
                # Reducir learning rate
                old_lr = self.learning_rate
                self.learning_rate = max(self.learning_rate * self.lr_factor, self.lr_min)
                if self.optimizer == "adam":
                    self.alpha = max(self.alpha * self.lr_factor, self.lr_min)
                
                if self.learning_rate != old_lr:
                    print(f"  ↓ Learning rate reducido a: {self.learning_rate}")
                
                self.lr_reduce_counter = 0

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

        # Delta de la última capa usando la función de pérdida seleccionada
        last_layer = len(self.weights) - 1
        loss_derivative = self.compute_loss_derivative(activations[-1], expected_output)
        deltas[last_layer] = loss_derivative * np.array([self.activator_derivative(z) for z in hidden_states[-1]])

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

        print(f"Iniciando entrenamiento con {epochs} épocas...")
        print(f"Optimizador: {self.optimizer}")
        print(f"Learning rate inicial: {self.initial_learning_rate}")
        print(f"Learning rate adaptativo: {'Sí' if self.adaptive_lr else 'No'}")
        print(f"Función de pérdida: {self.loss_function}")
        print(f"Arquitectura: {self.layers}")
        print("-" * 50)

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

                # Usar la función de pérdida seleccionada
                error += self.compute_loss(output, y)

            average_error = error / len(training_set)
            error_history.append(average_error)

            # Imprimir estado cada 1000 épocas o en épocas especiales
            if epoch % 1000 == 0 or epoch < 10 or epoch == epochs - 1:
                current_lr = self.learning_rate if self.optimizer != "adam" else self.alpha
                print(f"Época {epoch + 1:5d}/{epochs} | Error: {average_error:.6f} | Mejor: {best_error:.6f} | LR: {current_lr:.6f}")

            if average_error < (best_error - min_delta):
                best_error = average_error
                self.best_weights = copy.deepcopy(self.weights)
                self.patience_counter = 0
                if epoch % 1000 != 0:  # Evitar duplicar mensaje si ya se imprimió arriba
                    current_lr = self.learning_rate if self.optimizer != "adam" else self.alpha
                    print(f"Época {epoch + 1:5d}/{epochs} | Error: {average_error:.6f} | Mejor: {best_error:.6f} | LR: {current_lr:.6f} | ¡NUEVO MEJOR!")
            else:
                self.patience_counter += 1

                if len(error_history) >= window_size:
                    recent_errors = error_history[-window_size:]
                    if all(recent_errors[i] > recent_errors[i - 1] * 1.01 for i in range(1, len(recent_errors))):
                        print(f"Early stopping en época {epoch + 1} - Error aumenta consistentemente en {window_size} épocas")
                        break

                if self.patience_counter >= self.patience:
                    print(f"Early stopping en época {epoch + 1} - Paciencia agotada después de {self.patience} épocas sin mejora")
                    break

            self.adjust_learning_rate(average_error, best_error)

        print("-" * 50)
        print(f"Entrenamiento completado. Mejor error alcanzado: {best_error:.6f}")
        self.weights = copy.deepcopy(self.best_weights)

    def test(self, input_data):
        hidden_states, activations = self.forward_propagation(input_data)
        return activations[-1].tolist()

    def compute_loss(self, predicted, target):
        """Calcula la función de pérdida según la función seleccionada"""
        predicted = np.array(predicted)
        target = np.array(target)
        
        if self.loss_function == "mse":
            return np.mean((target - predicted) ** 2)
        elif self.loss_function == "binary_crossentropy":
            # Evitar log(0) y log(1) con epsilon
            epsilon = 1e-15
            predicted = np.clip(predicted, epsilon, 1 - epsilon)
            return -np.mean(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
        else:
            raise ValueError(f"Función de pérdida '{self.loss_function}' no soportada")

    def compute_loss_derivative(self, predicted, target):
        """Calcula la derivada de la función de pérdida"""
        predicted = np.array(predicted)
        target = np.array(target)
        
        if self.loss_function == "mse":
            return predicted - target
        elif self.loss_function == "binary_crossentropy":
            # Evitar división por cero
            epsilon = 1e-15
            predicted = np.clip(predicted, epsilon, 1 - epsilon)
            return (predicted - target) / (predicted * (1 - predicted))
        else:
            raise ValueError(f"Función de pérdida '{self.loss_function}' no soportada")
