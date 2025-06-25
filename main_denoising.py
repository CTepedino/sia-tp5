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

def pixel_difference(y_true, y_pred):
    total_dif = 0
    for true, pred in zip(y_true, y_pred):
        total_dif += np.sum(np.abs(np.array(true) - np.array(pred)))
    return total_dif

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return np.mean((y_true - y_pred) ** 2)


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

noise_error_functions = {
    "pixel": pixel_difference,
    "gaussian_binarized": pixel_difference,
    "gaussian": mean_squared_error
}

noise_error_unit = {
    "pixel": "píxeles",
    "gaussian_binarized": "píxeles",
    "gaussian": "MSE"
}

noise_level_unit_multiplier = {
    "pixel": 1,
    "gaussian_binarized": 0.1,
    "gaussian": 0.1
}

should_binarize_output = {
    "pixel": True,
    "gaussian_binarized": True,
    "gaussian": False
}

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)

    results_directory = "results/result_denoising_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs("results", exist_ok=True)
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
    noise_err_function = noise_error_functions[config["noise_function"]]
    noise_lv_multiplier = noise_level_unit_multiplier[config["noise_function"]]
    noise_err_unit = noise_error_unit[config["noise_function"]]
    binarize_output = should_binarize_output[config["noise_function"]]

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


    # dataset base
    plot_all_letters(
        np.array(letters),
        results_directory,
        filename="letter_map_original.png",
        titulo="Datos originales sin ruido"
    )

    with open(os.path.join(results_directory, "result_log.txt"), "w") as f:
        #genero datasets con distintos niveles de ruido y pruebo el dae
        for test_noise_lv in range(0, 11):
            next_noise = noise_lv_multiplier * test_noise_lv #pixeles -> [0 a 10], gauss -> [0.0 a 1]

            noisy_letters = noise_fn(letters, next_noise)

            plot_all_letters(noisy_letters, results_directory, filename=f"noisy_letters_{next_noise:.2f}.png", titulo=f"Conjunto de letras con ruido {next_noise}")

            errores_por_letra = []
            reconstructions = []
            for idx, letter in enumerate(letters):

                reconstruida = dae.test(noisy_letters[idx])

                if binarize_output:
                    reconstruida = (np.array(reconstruida) >= 0.5).astype(np.uint8)

                reconstructions.append(reconstruida)

                error_letra = noise_err_function(letters[idx], reconstruida)

                errores_por_letra.append(error_letra)

                log_and_print(f"Letra {font3_chars[idx]}: Error: {error_letra}", f)

            log_and_print(f"Error máximo por letra: {max(errores_por_letra)}", f)
            log_and_print(f"Error promedio por letra: {np.mean(errores_por_letra):.6f}", f)

            plot_all_letters(np.array(reconstructions), results_directory, filename=f"reconstructions_{next_noise:.2f}.png", titulo=f"Conjunto de reconstrucciones con ruido {next_noise}")

            error_counts = Counter(errores_por_letra)

            plt.figure(figsize=(10, 6))
            errors = sorted(error_counts.keys())
            counts = [error_counts[error] for error in errors]

            plt.bar(errors, counts)
            plt.xlabel(f'Error {noise_err_unit}')
            plt.ylabel('Cantidad de letras')
            plt.title('Distribución de errores por letra')
            plt.xticks(errors)
            plt.grid(True, alpha=0.3)

            # Agregar etiquetas de valor en las barras
            for i, (error, count) in enumerate(zip(errors, counts)):
                plt.text(error, count + 0.1, str(count), ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(results_directory, f"distribucion_errores_{next_noise:.2f}.png"))
            plt.show()


    # Visualizamos espacio latente
    z_list = []
    for letra_r in letters:
        _, activaciones = dae.forward_propagation(letra_r)
        z_list.append(activaciones[len(layers) // 2])

    z = np.array(z_list)
    plt.scatter(z[:, 0], z[:, 1])
    for i in range(len(z)):
        plt.annotate(font3_chars[i], (z[i, 0], z[i, 1]))
    plt.title("Representación en el espacio latente")
    plt.grid(True)
    plt.savefig(os.path.join(results_directory, "espacio_latente.png"))
    plt.show()






