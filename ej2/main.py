import numpy as np
import matplotlib.pyplot as plt

import vectorizedActivatorFunctions as activators
from emojis import emoji_images
from vae_layers import Layer, LatentLayer
from variational_autoencoder import MultiLayerPerceptron, VariationalAutoencoder


INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
LATENT_SIZE = 20
HIDDEN_SIZE = 100
HIDDEN_SIZE2 = 200
HIDDEN_SIZE3 = 300

EMOJIS_CHOSEN = len(emoji_images)

def graph_fonts(original, decoded):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')
    ax1.imshow(np.array(original).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.show()



if __name__ == "__main__":
    emoji_indexes = np.array([0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 23, 24, 26,
                              28, 29, 31, 32, 33, 35, 36, 38, 39, 41, 46, 48, 50, 51, 54, 55,
                              57, 58, 59, 61, 62, 63, 64, 65, 67, 73, 75, 78, 81, 83, 84, 85,
                              90, 91, 92, 93, 95])

    data = np.array(emoji_images)
    dataset_input = data[emoji_indexes]
    dataset_input_list = list(dataset_input)

    learning_rate = 0.0001

    main_activator = activators.activator_functions["relu"]
    last_activator = activators.activator_functions["sigmoid"]

    encoder = MultiLayerPerceptron()
    encoder.add_layer(Layer(input_dim=INPUT_SIZE, output_dim=HIDDEN_SIZE3, activator_function=main_activator[0], activator_derivative=main_activator[1],  learning_rate = learning_rate))
    encoder.add_layer(Layer(input_dim=HIDDEN_SIZE3, output_dim=HIDDEN_SIZE2,  activator_function=main_activator[0], activator_derivative=main_activator[1], learning_rate = learning_rate))
    encoder.add_layer(Layer(input_dim=HIDDEN_SIZE2, output_dim=HIDDEN_SIZE, activator_function=main_activator[0], activator_derivative=main_activator[1], learning_rate = learning_rate))

    sampler = LatentLayer(HIDDEN_SIZE, LATENT_SIZE, learning_rate=learning_rate)

    decoder = MultiLayerPerceptron()
    decoder.add_layer(Layer(input_dim=LATENT_SIZE, output_dim=HIDDEN_SIZE, activator_function=main_activator[0], activator_derivative=main_activator[1], learning_rate = learning_rate))
    decoder.add_layer(Layer(input_dim=HIDDEN_SIZE, output_dim=HIDDEN_SIZE2, activator_function=main_activator[0], activator_derivative=main_activator[1], learning_rate = learning_rate))
    decoder.add_layer(Layer(input_dim=HIDDEN_SIZE2, output_dim=HIDDEN_SIZE3, activator_function=main_activator[0], activator_derivative=main_activator[1], learning_rate = learning_rate))
    decoder.add_layer(Layer(input_dim=HIDDEN_SIZE3, output_dim=INPUT_SIZE, activator_function=last_activator[0], activator_derivative=last_activator[1], learning_rate = learning_rate))

    vae = VariationalAutoencoder(encoder, sampler, decoder)

    vae.train(dataset_input=dataset_input_list, epochs=100)

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output = vae.feedforward(input_reshaped)

        if i < 15:
            graph_fonts(list(dataset_input)[i], output)


    for _ in range(15):

        n = 10
        digit_size = INPUT_ROWS
        images = np.zeros((INPUT_ROWS, INPUT_COLS * n))

        random_index1 = np.random.choice(emoji_indexes)
        input_reshaped1 = np.reshape(emoji_images[random_index1], (len(emoji_images[random_index1]), 1))
        vae.feedforward(input_reshaped1)
        img1 = vae.sampler.sample

        random_index2 = np.random.choice(emoji_indexes)
        while random_index1 == random_index2:
            random_index2 = np.random.choice(emoji_indexes)
        input_reshaped2 = np.reshape(emoji_images[random_index2], (len(emoji_images[random_index2]), 1))
        vae.feedforward(input_reshaped2)
        img2 = vae.sampler.sample

        for i in range(n):
            z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
            output = vae.decoder.feedforward(z)
            output = output.reshape(INPUT_ROWS, INPUT_COLS)
            images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

        plt.figure(figsize=(10, 10))
        plt.imshow(images, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
