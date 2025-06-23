import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import vectorizedActivatorFunctions as activators
from ej2.emojis import emoji_images
from vae_layers import Layer, LatentLayer
from variational_autoencoder import MultiLayerPerceptron, VariationalAutoencoder

import seaborn as sns


import os
from datetime import datetime
import shutil

INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
# fijo, son los tamaños de los emojis


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


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

def save_emoji_grid(emojis, output_path, rows=None, cols=None):
    total = len(emojis)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))
    elif rows is None:
        rows = int(np.ceil(total / cols))
    elif cols is None:
        cols = int(np.ceil(total / rows))

    grid = np.zeros((rows * INPUT_ROWS, cols * INPUT_COLS))

    for idx, emoji in enumerate(emojis):
        r = idx // cols
        c = idx % cols
        image = emoji.reshape((INPUT_ROWS, INPUT_COLS))
        grid[r*INPUT_ROWS:(r+1)*INPUT_ROWS, c*INPUT_COLS:(c+1)*INPUT_COLS] = image

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(grid, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    latent_size = config["latent_size"]
    hidden_sizes = config["hidden_sizes"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    main_activator = activators.activator_functions[config["activator"]]
    last_activator = activators.activator_functions[config["output_activator"]]

    num_interpolations = config.get("num_interpolations", 15)
    interpolation_steps = config.get("interpolation_steps", 10)


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(sys.argv[1], os.path.join(output_dir, "config.json"))

    emoji_indexes = np.array(config.get("emoji_indexes", list(range(len(emoji_images)))))


    data = np.array(emoji_images)
    dataset_input = data[emoji_indexes]
    dataset_input_list = list(dataset_input)

    vae = build_vae(INPUT_SIZE, hidden_sizes, main_activator, learning_rate, latent_size, last_activator)

    vae.load_weights("vae_weights.npz")

    save_emoji_grid(dataset_input_list, os.path.join(output_dir, "emoji_grid.png"))

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output = vae.feedforward(input_reshaped)

        show_comparison(list(dataset_input)[i], output, i, output_dir)

    reconstructions = []

    for input_vec in dataset_input_list:
        input_reshaped = input_vec.reshape((-1, 1))
        output = vae.feedforward(input_reshaped)
        reconstructions.append(output.flatten())

    save_emoji_grid(reconstructions, os.path.join(output_dir, "vae_recon_grid.png"))

    for interpolation_idx in range(56):

        n = interpolation_steps
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

        fig, ax = plt.subplots(figsize=(n, 1.2))  # tamaño horizontal proporcional a n
        for i in range(n):
            alpha = i / (n - 1)
            z = (1 - alpha) * img1 + alpha * img2
            output = vae.decoder.feedforward(z)
            output = output.reshape(INPUT_ROWS, INPUT_COLS)
            images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

            # Anotar el valor de alpha o 'original'
            label = f"α = {alpha:.2f}"
            if i == 0:
                label = "original"
            elif i == n - 1:
                label = "original"
            ax.text(i * INPUT_COLS + INPUT_COLS / 2, -2, label,
                    ha='center', va='bottom', fontsize=8)

        ax.imshow(images, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"interpolation_{interpolation_idx}.png"))
        plt.close(fig)

    if latent_size == 2:
        # Obtener las coordenadas latentes mu para cada emoji
        latent_coords = []
        for emoji in dataset_input_list:
            input_reshaped = np.reshape(emoji, (INPUT_SIZE, 1))
            vae.feedforward(input_reshaped)
            mu = vae.latent.mu
            latent_coords.append(mu.flatten())

        latent_coords = np.array(latent_coords)

        # Gráfico del espacio latente
        plt.figure(figsize=(10,10))
        sns.scatterplot(x=latent_coords[:, 0], y=latent_coords[:, 1], hue=emoji_indexes, palette='tab10', s=100)
        plt.title("Espacio latente (2D) del VAE")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(title="Emoji index", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_space.png"))
        plt.close()

    # Calcular MSE por emoji
    mse_errors = []
    for i, emoji in enumerate(dataset_input_list):
        input_reshaped = np.reshape(emoji, (INPUT_SIZE, 1))
        output = vae.feedforward(input_reshaped)
        mse_errors.append( mse(emoji, output.flatten()))

    avg_mse = np.mean(mse_errors)

    plt.figure(figsize=(8, 5))
    plt.hist(mse_errors, bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribución del error de reconstrucción (MSE)")
    plt.xlabel("Error de reconstrucción (MSE)")
    plt.ylabel("Cantidad de emojis")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_mse_histogram.png"))
    plt.close()
