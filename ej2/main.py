import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import vectorizedActivatorFunctions as activators
from ej2.emojis import emoji_images
from vae_layers import Layer, LatentLayer
from variational_autoencoder import MultiLayerPerceptron, VariationalAutoencoder

import os
from datetime import datetime
import shutil

INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
# fijo, son los tamaños de los emojis

def show_comparison(original, decoded, index, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decodificado')
    ax1.imshow(np.array(original).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reconstruction_{index}.png"))
    plt.close(fig)

def build_encoder(input_size, hidden_sizes, activator, learning_rate):
    encoder = MultiLayerPerceptron()
    layer_dims = [input_size] + hidden_sizes
    for i in range(len(layer_dims) - 1):
        encoder.add_layer(Layer(
            input_dim=layer_dims[i],
            output_dim=layer_dims[i+1],
            activator_function=activator[0],
            activator_derivative=activator[1],
            learning_rate=learning_rate
        ))
    return encoder

def build_decoder(latent_size, hidden_sizes, activator, output_activator, learning_rate, output_size):
    decoder = MultiLayerPerceptron()
    layer_dims = [latent_size] + hidden_sizes[::-1] + [output_size]
    for i in range(len(layer_dims) - 2):
        decoder.add_layer(Layer(
            input_dim=layer_dims[i],
            output_dim=layer_dims[i+1],
            activator_function=activator[0],
            activator_derivative=activator[1],
            learning_rate=learning_rate
        ))
    # Output layer
    decoder.add_layer(Layer(
        input_dim=layer_dims[-2],
        output_dim=layer_dims[-1],
        activator_function=output_activator[0],
        activator_derivative=output_activator[1],
        learning_rate=learning_rate
    ))
    return decoder

def build_vae(INPUT_SIZE, hidden_sizes, main_activator, learning_rate, latent_size, last_activator):
    encoder = build_encoder(INPUT_SIZE, hidden_sizes, main_activator, learning_rate)
    latent = LatentLayer(hidden_sizes[-1], latent_size, learning_rate)
    decoder = build_decoder(latent_size, hidden_sizes, main_activator, last_activator, learning_rate, INPUT_SIZE)
    return VariationalAutoencoder(encoder, latent, decoder)

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    latent_size = config["latent_size"]
    hidden_sizes = config["hidden_sizes"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    main_activator = activators.activator_functions[config["activator"]]
    last_activator = activators.activator_functions[config["output_activator"]]


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config, os.path.join(output_dir, "config.json"))


    emoji_indexes = np.array([0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 23, 24, 26,
                              28, 29, 31, 32, 33, 35, 36, 38, 39, 41, 46, 48, 50, 51, 54, 55,
                              57, 58, 59, 61, 62, 63, 64, 65, 67, 73, 75, 78, 81, 83, 84, 85,
                              90, 91, 92, 93, 95])

    data = np.array(emoji_images)
    dataset_input = data[emoji_indexes]
    dataset_input_list = list(dataset_input)

    vae = build_vae(INPUT_SIZE, hidden_sizes, main_activator, learning_rate, latent_size, last_activator)

    vae.train(dataset_input=dataset_input_list, epochs=epochs)

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output = vae.feedforward(input_reshaped)

        show_comparison(list(dataset_input)[i], output, i, output_dir)


    for interpolation_idx in range(15):

        n = 10
        digit_size = INPUT_ROWS
        images = np.zeros((INPUT_ROWS, INPUT_COLS * n))

        random_index1 = np.random.choice(emoji_indexes)
        input_reshaped1 = np.reshape(emoji_images[random_index1], (len(emoji_images[random_index1]), 1))
        vae.feedforward(input_reshaped1)
        img1 = vae.latent.sample

        random_index2 = np.random.choice(emoji_indexes)
        while random_index1 == random_index2:
            random_index2 = np.random.choice(emoji_indexes)
        input_reshaped2 = np.reshape(emoji_images[random_index2], (len(emoji_images[random_index2]), 1))
        vae.feedforward(input_reshaped2)
        img2 = vae.latent.sample

        for i in range(n):
            z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
            output = vae.decoder.feedforward(z)
            output = output.reshape(INPUT_ROWS, INPUT_COLS)
            images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(images, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"interpolation_{interpolation_idx}.png"))
        plt.close(fig)
