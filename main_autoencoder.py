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

def load_config(config_path):
    """Carga la configuración desde un archivo JSON"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['layers', 'learning_rate', 'function', 'optimizer', 'loss_function', 'epochs', 'interpolation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta el parámetro '{key}' en la configuración")
        
        interpolation_keys = ['letra1_idx', 'letra2_idx', 'n_interpolations']
        for key in interpolation_keys:
            if key not in config['interpolation']:
                raise ValueError(f"Falta el parámetro '{key}' en la sección 'interpolation'")
        
        return config
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de configuración '{config_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: El archivo de configuración no es un JSON válido: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error en la configuración: {e}")
        sys.exit(1)

def font_to_binary_patterns():
    patterns = []
    for symbol in Font3:
        binary_pattern = []
        for row in symbol:
            row_bits = [(row >> (4 - bit)) & 1 for bit in range(5)]  # bits de izquierda a derecha
            binary_pattern.extend(row_bits)
        patterns.append(binary_pattern)
    return patterns


def contar_error_pixel(x, x_hat, umbral=0.5):
    errores = 0
    for orig, recon in zip(x, x_hat):
        errores += sum(abs(np.array(orig) - (np.array(recon) > umbral).astype(int)))
    return errores

def log_and_print(msg, file):
    print(msg)
    file.write(str(msg) + '\n')

def plot_all_letters(data, results_directory):
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
    fig.savefig(os.path.join(results_directory, "letter_map.png"))
    plt.show()

def generate_new_letters_from_latent_space(ae, letras, results_directory, capa_latente, config):
    """
    Genera nuevas letras interpolando en el espacio latente entre letras existentes
    """
    print("\n" + "="*60)
    print("GENERANDO NUEVAS LETRAS DESDE EL ESPACIO LATENTE")
    print("="*60)
    
    letra1_idx = config['interpolation']['letra1_idx']
    letra2_idx = config['interpolation']['letra2_idx']
    n_interpolations = config['interpolation']['n_interpolations']
    
    if letra1_idx < 0 or letra1_idx >= len(letras) or letra2_idx < 0 or letra2_idx >= len(letras):
        print(f"Error: Los índices de letras deben estar entre 0 y {len(letras)-1}")
        return
    
    z_list = []
    for letra in letras:
        _, activaciones = ae.forward_propagation(letra)
        z_list.append(activaciones[capa_latente])
    
    z = np.array(z_list)
    
    def generate_from_latent_point(latent_point):
        """Genera una letra desde un punto en el espacio latente usando solo el decoder"""
        current_activation = np.array(latent_point)
        
        decoder_start = capa_latente
        
        for layer_idx in range(decoder_start, len(ae.weights)):
            current_activation_with_bias = np.append(current_activation, 1.0)
            layer_weights = np.array(ae.weights[layer_idx])
            
            h = np.dot(layer_weights, current_activation_with_bias)
            current_activation = np.array([ae.activator_function(x) for x in h])
        
        return current_activation.tolist()
    
    print(f"\nInterpolación lineal entre letras '{font3_chars[letra1_idx]}' y '{font3_chars[letra2_idx]}':")
    
    z1, z2 = z[letra1_idx], z[letra2_idx]
    
    interpolated_letters = []
    
    fig, axs = plt.subplots(1, n_interpolations + 2, figsize=(15, 3))
    
    letra1_bin = (np.array(letras[letra1_idx]) > 0.5).astype(int)
    axs[0].imshow(letra1_bin.reshape(7, 5), cmap="binary")
    axs[0].set_title(f"'{font3_chars[letra1_idx]}' original")
    axs[0].axis("off")
    
    for i in range(n_interpolations):
        alpha = (i + 1) / (n_interpolations + 1)
        interpolated_z = z1 * (1 - alpha) + z2 * alpha
        
        generated_letter = generate_from_latent_point(interpolated_z)
        generated_bin = (np.array(generated_letter) > 0.5).astype(int)
        interpolated_letters.append(generated_bin)
        
        axs[i + 1].imshow(generated_bin.reshape(7, 5), cmap="binary")
        axs[i + 1].set_title(f"α={alpha:.2f}")
        axs[i + 1].axis("off")
    
    letra2_bin = (np.array(letras[letra2_idx]) > 0.5).astype(int)
    axs[-1].imshow(letra2_bin.reshape(7, 5), cmap="binary")
    axs[-1].set_title(f"'{font3_chars[letra2_idx]}' original")
    axs[-1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "interpolacion_lineal.png"))
    plt.show()
    
    with open(os.path.join(results_directory, "letras_interpoladas.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Interpolación", "Alpha", "Bits generados"])
        
        for i, letra in enumerate(interpolated_letters):
            alpha = (i + 1) / (n_interpolations + 1)
            bits_str = ''.join(str(bit) for bit in letra)
            writer.writerow([i+1, f"{alpha:.2f}", bits_str])


def entrenar_autoencoder(results_directory, config):
    letras = font_to_binary_patterns()
    activador, activador_deriv = non_linear_functions[config["function"]]

    params = {
        "layers": config["layers"],
        "learning_rate": config["learning_rate"],
        "function": config["function"],
        "optimizer": config["optimizer"],
        "loss_function": config["loss_function"],
        "epochs": config["epochs"]
    }

    capa_latente = len(params["layers"]) // 2
    print(f"Capa latente: {capa_latente}")
    
    with open(os.path.join(results_directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    arch_str = '-'.join(str(x) for x in params["layers"])
    weights_path = os.path.join(weights_dir, f"MLP_{arch_str}_autoencoder.npy")

    ae = MultiLayerPerceptron(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        activator_function=activador,
        activator_derivative=activador_deriv,
        optimizer=params["optimizer"],
        loss_function=params["loss_function"]
    )

    if os.path.exists(weights_path):
        print(f"Cargando pesos desde: {weights_path}")
        ae.weights = list(np.load(weights_path, allow_pickle=True))
        print("Pesos cargados correctamente. Se continuará el entrenamiento.")
    else:
        print("No se encontraron pesos previos. Se entrenará desde cero.")

    ae.train(letras, letras, epochs=config["epochs"])
    np.save(weights_path, np.array(ae.weights, dtype=object))
    print(f"Pesos guardados en: {weights_path}")

    with open(os.path.join(results_directory, "result.txt"), "w") as f:
        errores_por_letra = []
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letra)
            error_letra = sum(abs(np.array(letra) - (np.array(reconstruida) > 0.5).astype(int)))
            errores_por_letra.append(error_letra)
            log_and_print(f"Letra {font3_chars[idx]}: Error: {error_letra}", f)

        log_and_print(f"Error máximo por letra: {max(errores_por_letra)}", f)
        log_and_print(f"Error promedio por letra: {np.mean(errores_por_letra):.6f}", f)

        error_counts = Counter(errores_por_letra)
        
        plt.figure(figsize=(10, 6))
        errors = sorted(error_counts.keys())
        counts = [error_counts[error] for error in errors]
        
        plt.bar(errors, counts)
        plt.xlabel('Error (píxeles)')
        plt.ylabel('Cantidad de letras')
        plt.title('Distribución de errores por letra')
        plt.xticks(errors)
        plt.grid(True, alpha=0.3)
        
        for i, (error, count) in enumerate(zip(errors, counts)):
            plt.text(error, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, "distribucion_errores.png"))
        plt.show()

        z_list = []
        for letra in letras:
            _, activaciones = ae.forward_propagation(letra)
            z_list.append(activaciones[capa_latente])

        z = np.array(z_list)
        plt.scatter(z[:, 0], z[:, 1])
        for i in range(len(z)):
            plt.annotate(font3_chars[i], (z[i, 0], z[i, 1]))
        plt.title("Representación en el espacio latente (2D)")
        plt.grid(True)
        plt.savefig(os.path.join(results_directory, "espacio_latente.png"))
        plt.show()

    letras_reconstruidas = []
    with open(os.path.join(results_directory, "resultado_letras.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Letra", "Bits reconstruidos"])
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letra)
            reconstruida_bin = (np.array(reconstruida) > 0.5).astype(int)
            letras_reconstruidas.append(reconstruida_bin)
            bits_str = ''.join(str(bit) for bit in reconstruida_bin)
            writer.writerow([font3_chars[idx], bits_str])

    plot_all_letters(np.array(letras_reconstruidas), results_directory)

    generate_new_letters_from_latent_space(ae, letras, results_directory, capa_latente, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar autoencoder y generar nuevas letras')
    parser.add_argument('config', help='Ruta al archivo de configuración JSON')
    parser.add_argument('--output-dir', '-o', help='Directorio de salida (opcional, por defecto se crea automáticamente)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.output_dir:
        results_directory = args.output_dir
    else:
        results_directory = "results/result_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    os.makedirs(results_directory, exist_ok=True)
    
    print("Configuración cargada:")
    print(f"- Arquitectura: {config['layers']}")
    print(f"- Learning rate: {config['learning_rate']}")
    print(f"- Función de activación: {config['function']}")
    print(f"- Optimizador: {config['optimizer']}")
    print(f"- Función de pérdida: {config['loss_function']}")
    print(f"- Épocas: {config['epochs']}")
    print(f"- Interpolación: letra {config['interpolation']['letra1_idx']} ({font3_chars[config['interpolation']['letra1_idx']]}) a letra {config['interpolation']['letra2_idx']} ({font3_chars[config['interpolation']['letra2_idx']]})")
    print(f"- Número de interpolaciones: {config['interpolation']['n_interpolations']}")
    print(f"- Directorio de resultados: {results_directory}")
    print("-" * 60)
    
    entrenar_autoencoder(results_directory, config)
