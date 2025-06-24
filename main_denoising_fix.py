import numpy as np
from multilayer_perceptron import MultiLayerPerceptron
import matplotlib.pyplot as plt
from activatorFunctions import non_linear_functions
import os
from datetime import datetime
import json
from collections import Counter
import csv
import sys
import argparse


# El Font3 original en decimal (cada número representa una fila de 5 bits)
Font3 = [
    [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],
    [0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d, 0x00],
    [0x10, 0x10, 0x10, 0x1c, 0x12, 0x12, 0x1c],
    [0x00, 0x00, 0x00, 0x0e, 0x10, 0x10, 0x0e],
    [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],
    [0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0f],
    [0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08],
    [0x0e, 0x11, 0x13, 0x0d, 0x01, 0x01, 0x0e],
    [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],
    [0x00, 0x04, 0x00, 0x0c, 0x04, 0x04, 0x0e],
    [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c],
    [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
    [0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    [0x00, 0x00, 0x0a, 0x15, 0x15, 0x11, 0x11],
    [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
    [0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e],
    [0x00, 0x1c, 0x12, 0x12, 0x1c, 0x10, 0x10],
    [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],
    [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
    [0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e],
    [0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06],
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d],
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04],
    [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a],
    [0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11],
    [0x00, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e],
    [0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f],
    [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],
    [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],
    [0x0c, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0c],
    [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],
    [0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f]
]
font3_chars = [
    "`", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "{", "|", "}", "~", "DEL"
]

def log_and_print(msg, file):
    print(msg)
    file.write(str(msg) + '\n')

# Convertir Font3 a vectores binarios de 35 bits
def font_to_binary_patterns():
    patterns = []
    for symbol in Font3:
        binary_pattern = []
        for row in symbol:
            row_bits = [(row >> (4 - bit)) & 1 for bit in range(5)]
            binary_pattern.extend(row_bits)
        patterns.append(np.array(binary_pattern))
    return patterns

def plot_all_letters(data, results_directory, filename=None, titulo=None):
    n_letters = len(data)
    n_cols = 8
    n_rows = (n_letters + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axs = axs.flatten()

    for i in range(n_letters):
        letter = data[i].reshape(7, 5)
        axs[i].imshow(letter, cmap="binary")
        axs[i].axis("off")

    fig.tight_layout()
    if titulo:
        fig.suptitle(titulo, fontsize=16)
        plt.subplots_adjust(top=0.88)
    if filename:
        fig.savefig(os.path.join(results_directory, filename))
    else:
        fig.savefig(os.path.join(results_directory, "letter_map.png"))
    plt.show()

def pixel_noise(letters, n_pixels):
    noisy_letters = []
    for letter in letters:
        noisy = np.array(letter).copy()
        indices = np.random.choice(len(noisy), size=n_pixels, replace=False)
        noisy[indices] = 1 - noisy[indices]
        noisy_letters.append(noisy)
    return noisy_letters

def gaussian_noise_binarized(letters, std):
    noisy_letters = []
    for letter in letters:
        letter = np.array(letter, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=std, size=letter.shape)
        noisy = letter + noise
        noisy = np.clip(noisy, 0.0, 1.0)
        noisy = (noisy >= 0.5).astype(np.uint8)
        noisy_letters.append(noisy)
    return noisy_letters


def gaussian_noise(letters, std):
    noisy_letters = []
    for letter in letters:
        letter = np.array(letter, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=std, size=letter.shape)
        noisy = letter + noise
        noisy = np.clip(noisy, 0.0, 1.0)
        noisy_letters.append(noisy)
    return noisy_letters

noise_functions = {
    "pixel": pixel_noise,
    "gaussian": gaussian_noise,
    "gaussian_binarized": gaussian_noise_binarized
}

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)

    results_directory = "results/result_denoising_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(results_directory, exist_ok=True)

    with open(os.path.join(results_directory, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    #Cargo parametros y dataset

    letters = font_to_binary_patterns()

    activator, activator_derivative = non_linear_functions[config["function"]]

    learning_rate = config["learning_rate"]

    epochs = config["epochs"]

    optimizer = config["optimizer"]

    loss_fn = config["loss_function"]

    hidden_layers = config["hidden_layers"]
    latent_size = config["latent_size"]
    layers = hidden_layers + [latent_size] + hidden_layers[::-1]

    noise_fn = noise_functions[config["noise_function"]]
    noise_level = config["noise_level"]

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    arch_str = '-'.join(str(x) for x in layers)
    weights_path = os.path.join(weights_dir, f"MLP_{arch_str}_denoising.npy")

    # Creo y entreno el DAE

    dae = MultiLayerPerceptron(
        layers=layers,
        learning_rate=learning_rate,
        activator_function=activator,
        activator_derivative=activator_derivative,
        optimizer=optimizer,
        loss_function=loss_fn,

        noise_function=noise_fn,
        noise_level=noise_level
    )

    dae.train(letters, letters, epochs=epochs)

    np.save(weights_path, np.array(dae.weights, dtype=object))

    # Graficos, analisis, etc





