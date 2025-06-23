import numpy as np

def mse(y, a):
    temp = a - y
    return (1.0 / (2.0 * y.shape[1])) * np.sum(temp ** 2)


def mse_derivative(y, a):
    # return (a - y) / y.shape[1]
    return (1. / y.shape[1]) * np.sum(a - y, axis=1).reshape((y.shape[0],1))

class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feedforward(self, input_data):
        for layer in self.layers:
            input_data = layer.feedforward(input_data)
        return input_data

    def predict(self, input_data):
        input_data = input_data.reshape((input_data.shape[0], 1))
        return self.feedforward(input_data)

    def backpropagate(self, output, use_loss=True):
        if use_loss:

            last_gradient = mse_derivative(output, self.layers[-1].a) * \
                self.layers[-1].activator_derivative(self.layers[-1].z)


            is_output_layer = True
            for layer in reversed(self.layers):
                last_gradient = layer.backward(
                    last_gradient,
                    output_layer=is_output_layer,
                )
                is_output_layer = False
        else:
            last_gradient = output
            for layer in reversed(self.layers):
                last_gradient = layer.backward(
                    last_gradient,
                    output_layer=False,
                )

    def train(self, dataset_input, dataset_output, epochs=1, batch_size=1):
        for layer in self.layers:
            layer.set_batch_size(batch_size)

        samples_processed = 0
        for epoch in range(epochs):
            for i in range(len(dataset_input)):
                input_batch = np.reshape(dataset_input[i], (len(dataset_input[i]), batch_size))
                output_batch = np.reshape(dataset_output[i], (len(dataset_output[i]), batch_size))

                self.feedforward(input_batch)
                self.backpropagate(output_batch)

                samples_processed += batch_size


class VariationalAutoencoder(MultiLayerPerceptron):
    def __init__(self, encoder, sampler, decoder):
        super().__init__()

        self.layers = encoder.layers + [sampler.mean, sampler.log_var] + decoder.layers
        self.encoder = encoder
        self.sampler = sampler
        self.decoder = decoder

    def feedforward(self, input_data):
        encoder_output = self.encoder.feedforward(input_data)
        sample = self.sampler.feedforward(encoder_output)
        decoder_output = self.decoder.feedforward(sample)
        return decoder_output

    def backpropagate(self, target_output, use_loss=False):
        self.decoder.backpropagate(target_output)
        decoder_gradient = self.decoder.layers[0].gradient
        sampler_gradient = self.sampler.backpropagate(decoder_gradient)
        self.encoder.backpropagate(sampler_gradient, use_loss=False)

    def train(self, dataset_input, dataset_test=None, epochs=1, batch_size=1):
        super().train(dataset_input=dataset_input, dataset_output=dataset_input, epochs=epochs, batch_size=batch_size)

