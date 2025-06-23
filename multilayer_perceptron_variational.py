import numpy as np
import copy


class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative=lambda x: 1,
                 optimizer="gradient", split_output = None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.split_output = split_output

        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        self.weights = []
        for i in range(len(layers) - 1):
            neurons = layers[i + 1]
            inputs = layers[i] + 1
            scale = np.sqrt(2.0 / (inputs + neurons))
            self.weights.append(np.random.normal(0, scale, (neurons, inputs)).tolist())

        self.min_error = None
        self.best_weights = None
        self.patience = 50
        self.patience_counter = 0

        if self.optimizer == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = learning_rate
            self.m = [np.zeros_like(np.array(w)) for w in self.weights]
            self.v = [np.zeros_like(np.array(w)) for w in self.weights]
            self.t = 0

            if len(layers) <= 3 and max(layers) <= 10:
                self.beta1 = 0.8
                self.beta2 = 0.9
                self.epsilon = 1e-6
                self.alpha = learning_rate * 0.1

        elif self.optimizer == "momentum":
            self.momentum = 0.9
            self.velocity = [np.zeros_like(np.array(w)) for w in self.weights]

    def update_weights_gradient(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        self.weights[l] = np.array(self.weights[l]) - self.learning_rate * weight_gradients

    def update_weights_momentum(self, l, delta, activation):
        weight_gradients = np.outer(delta, activation)
        self.velocity[l] = self.momentum * self.velocity[l] - self.learning_rate * weight_gradients
        self.weights[l] = np.array(self.weights[l]) + self.velocity[l]

    def update_weights_adam(self, l, delta, activation):
        self.t += 1
        weight_gradients = np.outer(delta, activation)
        self.m[l] = self.beta1 * self.m[l] + (1 - self.beta1) * weight_gradients
        self.v[l] = self.beta2 * self.v[l] + (1 - self.beta2) * np.square(weight_gradients)
        m_hat = self.m[l] / (1 - self.beta1 ** self.t)
        v_hat = self.v[l] / (1 - self.beta2 ** self.t)
        update = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if len(self.layers) <= 3 and max(self.layers) <= 10:
            update = np.clip(update, -0.1, 0.1)

        self.weights[l] = np.array(self.weights[l]) - update

    def forward_propagation(self, input_data):
        activations = [np.array(input_data)]
        hidden_states = []

        for layer_index in range(len(self.weights)):
            prev_activation = np.append(activations[-1], 1.0)
            layer_weights = np.array(self.weights[layer_index])
            h = np.dot(layer_weights, prev_activation)
            a = np.array([self.activator_function(x) for x in h])

            hidden_states.append(h)
            activations.append(a)

        if self.split_output is not None and len(activations) > 1:
            final_output = activations[-1]
            size = len(final_output)
            if size % self.split_output != 0:
                raise ValueError("La cantidad de neuronas de salida no es divisible por split_output.")
            chunk_size = size // self.split_output
            split_outputs = [final_output[i * chunk_size: (i + 1) * chunk_size] for i in range(self.split_output)]
            return hidden_states, activations, split_outputs

        return hidden_states, activations

    def back_propagation(self, expected_output, hidden_states, activations):
        deltas = [None] * len(self.weights)
        expected_output = np.array(expected_output)

        last_layer = len(self.weights) - 1
        output_error = activations[-1] - expected_output
        deltas[last_layer] = output_error * np.array([self.activator_derivative(z) for z in hidden_states[-1]])

        for l in range(len(self.weights) - 2, -1, -1):
            next_weights = np.array(self.weights[l + 1])
            next_delta = deltas[l + 1]
            current_h = hidden_states[l]
            delta = np.dot(next_weights[:, :-1].T, next_delta) * np.array(
                [self.activator_derivative(z) for z in current_h])
            deltas[l] = delta

        for l in range(len(self.weights)):
            delta = deltas[l]
            activation = np.append(activations[l], 1.0)

            if self.optimizer == "adam":
                self.update_weights_adam(l, delta, activation)
            elif self.optimizer == "momentum":
                self.update_weights_momentum(l, delta, activation)
            else:
                self.update_weights_gradient(l, delta, activation)

    def train(self, training_set, expected_outputs, epochs):
        best_error = float('inf')
        error_history = []
        min_delta = 1e-5
        window_size = 10

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

                error += 0.5 * np.sum((np.array(y) - output) ** 2)

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


class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, hidden_layers, learning_rate,
                 activator_function, activator_derivative, optimizer="adam", kl_weight=0.01):
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = MultiLayerPerceptron(
            layers=[input_dim] + hidden_layers + [2 * latent_dim],
            learning_rate=learning_rate,
            activator_function=activator_function,
            activator_derivative=activator_derivative,
            optimizer=optimizer,
            split_output=2  # (mu, logvar)
        )

        self.decoder = MultiLayerPerceptron(
            layers=[latent_dim] + hidden_layers[::-1] + [input_dim],
            learning_rate=learning_rate,
            activator_function=activator_function,
            activator_derivative=activator_derivative,
            optimizer=optimizer
        )

    def reparameterize(self, mu, logvar):
        epsilon = np.random.normal(size=mu.shape)
        return mu + np.exp(0.5 * logvar) * epsilon, epsilon

    def loss(self, x_true, x_recon, mu, logvar):
        x_true = np.array(x_true)
        x_recon = np.array(x_recon)

        recon_loss = 0.5 * np.mean((x_true - x_recon) ** 2)
        kl_div = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))

        return recon_loss, kl_div

    def forward(self, x):
        _, _, (mu, logvar) = self.encoder.forward_propagation(x)
        mu = np.array(mu)
        logvar = np.clip(np.array(logvar), -10, 10)
        z, epsilon = self.reparameterize(mu, logvar)
        _, activations = self.decoder.forward_propagation(z)
        x_recon = activations[-1]
        return x_recon, mu, logvar, z, epsilon

    def train(self, dataset, epochs=100, verbose=True, max_kl_weight=0.001, warmup_epochs=30):
        self.loss_history = []
        self.kl_history = []
        self.recon_history = []

        for epoch in range(epochs):
            kl_weight = min(max_kl_weight, epoch / warmup_epochs * max_kl_weight)

            total_loss = 0
            total_kl = 0
            total_recon = 0

            for x in dataset:
                x_recon, mu, logvar, z, epsilon = self.forward(x)
                recon_loss, kl_div = self.loss(x, x_recon, mu, logvar)

                loss = recon_loss + kl_weight * kl_div
                total_loss += loss
                total_recon += recon_loss
                total_kl += kl_div

                # === BACKPROP ===
                self.decoder.back_propagation(x, *self.decoder.forward_propagation(z)[:2])

                hidden_states, activations = self.decoder.forward_propagation(z)
                error = activations[-1] - np.array(x)
                delta = error * np.array([self.decoder.activator_derivative(h) for h in hidden_states[-1]])

                for l in reversed(range(len(self.decoder.weights))):
                    W = np.array(self.decoder.weights[l])[:, :-1]
                    if l == 0:
                        dz = W.T @ delta
                        break
                    h = hidden_states[l - 1]
                    delta = W.T @ delta * np.array([self.decoder.activator_derivative(hh) for hh in h])

                dmu = dz + mu
                dlogvar = dz * epsilon * 0.5 * np.exp(0.5 * logvar) + 0.5 * (np.exp(logvar) - 1)

                target = np.concatenate([mu - kl_weight * dmu, logvar - kl_weight * dlogvar])
                self.encoder.back_propagation(target, *self.encoder.forward_propagation(x)[:2])

            n = len(dataset)
            avg_loss = total_loss / n
            avg_recon = total_recon / n
            avg_kl = total_kl / n

            self.loss_history.append(avg_loss)
            self.recon_history.append(avg_recon)
            self.kl_history.append(avg_kl)

            if verbose:
                print(
                    f"[Epoch {epoch + 1:3}] Total: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KL: {avg_kl:.6f} | Weight: {kl_weight:.5f}")

    def encode(self, x):
        _, _, (mu, logvar) = self.encoder.forward_propagation(x)
        return np.array(mu), np.array(logvar)

    def decode(self, z):
        _, activations = self.decoder.forward_propagation(z)
        return activations[-1]

    def generate(self, n_samples=1):
        return [self.decode(np.random.normal(0, 1, self.latent_dim)) for _ in range(n_samples)]
