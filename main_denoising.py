import numpy as np
from multilayer_perceptron_denoising import MultiLayerPerceptron
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

def load_config(config_path):
    """Carga la configuración desde un archivo JSON"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validar configuración
        required_keys = ['layers', 'learning_rate', 'function', 'optimizer', 'loss_function', 'epochs', 'n_pixeles_ruido']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta el parámetro '{key}' en la configuración")
        
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

# Convertir Font3 a vectores binarios de 35 bits
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

def agregar_ruido(letras, n_pixeles=3):
    letras_ruidosas = []
    for letra in letras:
        letra_r = np.array(letra).copy()
        indices = np.random.choice(len(letra_r), size=n_pixeles, replace=False)
        letra_r[indices] = 1 - letra_r[indices]  # Flip bit
        letras_ruidosas.append(letra_r)
    return letras_ruidosas

def entrenar_autoencoder(results_directory, config):
    letras = font_to_binary_patterns()
    letras_ruidosas = agregar_ruido(letras, n_pixeles=config["n_pixeles_ruido"])
    # Elegir la función de activación por nombre
    activador, activador_deriv = non_linear_functions[config["function"]]

    # Guardar parámetros en JSON
    params = {
        "layers": config["layers"],
        "learning_rate": config["learning_rate"],
        "function": config["function"],
        "optimizer": config["optimizer"],
        "loss_function": config["loss_function"],
        "epochs": config["epochs"],
        "n_pixeles_ruido": config["n_pixeles_ruido"]
    }

    capa_latente = len(params["layers"]) // 2
    print(f"Capa latente: {capa_latente}")
    
    with open(os.path.join(results_directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Guardar/cargar los pesos en la carpeta 'weights/'
    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    arch_str = '-'.join(str(x) for x in params["layers"])
    weights_path = os.path.join(weights_dir, f"MLP_{arch_str}_denoising.npy")

    # Crear el modelo
    ae = MultiLayerPerceptron(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        activator_function=activador,
        activator_derivative=activador_deriv,
        optimizer=params["optimizer"],
        loss_function=params["loss_function"]
    )

    # Intentar cargar pesos
    if os.path.exists(weights_path):
        print(f"Cargando pesos desde: {weights_path}")
        ae.weights = list(np.load(weights_path, allow_pickle=True))
        print("Pesos cargados correctamente. Se continuará el entrenamiento.")
    else:
        print("No se encontraron pesos previos. Se entrenará desde cero.")

    # Siempre entrenar (continuar o desde cero)
    ae.train(letras_ruidosas, letras, epochs=config["epochs"])
    np.save(weights_path, np.array(ae.weights, dtype=object))
    print(f"Pesos guardados en: {weights_path}")

    with open(os.path.join(results_directory, "result.txt"), "w") as f:
        errores_por_letra = []
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letras_ruidosas[idx])
            error_letra = sum(abs(np.array(letra) - (np.array(reconstruida) > 0.5).astype(int)))
            errores_por_letra.append(error_letra)
            log_and_print(f"Letra {font3_chars[idx]}: Error: {error_letra}", f)

        log_and_print(f"Error máximo por letra: {max(errores_por_letra)}", f)
        log_and_print(f"Error promedio por letra: {np.mean(errores_por_letra):.6f}", f)

        # Gráfico de barras de distribución de errores
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
        
        # Agregar etiquetas de valor en las barras
        for i, (error, count) in enumerate(zip(errors, counts)):
            plt.text(error, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, "distribucion_errores.png"))
        plt.show()

        # Visualizamos espacio latente
        z_list = []
        for letra_r in letras_ruidosas:
            _, activaciones = ae.forward_propagation(letra_r)
            z_list.append(activaciones[capa_latente])

        z = np.array(z_list)
        plt.scatter(z[:, 0], z[:, 1])
        for i in range(len(z)):
            plt.annotate(font3_chars[i], (z[i, 0], z[i, 1]))
        plt.title("Representación en el espacio latente (2D)")
        plt.grid(True)
        plt.savefig(os.path.join(results_directory, "espacio_latente.png"))
        plt.show()

    # Guardar resultados de letras predichas
    letras_reconstruidas = []
    with open(os.path.join(results_directory, "resultado_letras.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Letra", "Bits reconstruidos"])
        for idx, letra_r in enumerate(letras_ruidosas):
            reconstruida = ae.test(letra_r)
            reconstruida_bin = (np.array(reconstruida) > 0.5).astype(int)
            letras_reconstruidas.append(reconstruida_bin)
            bits_str = ''.join(str(bit) for bit in reconstruida_bin)
            writer.writerow([font3_chars[idx], bits_str])

    # Graficar todas las letras reconstruidas
    plot_all_letters(np.array(letras_reconstruidas), results_directory, titulo="Resultado final")

    # Graficar todas las letras de entrada (con ruido)
    plot_all_letters(
        np.array(letras_ruidosas),
        results_directory,
        filename="letter_map_ruido.png",
        titulo=f"Entrada ruidosa con {config['n_pixeles_ruido']} píxeles modificados"
    )

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar autoencoder de denoising')
    parser.add_argument('config', help='Ruta al archivo de configuración JSON')
    parser.add_argument('--output-dir', '-o', help='Directorio de salida (opcional, por defecto se crea automáticamente)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Crear directorio de resultados
    if args.output_dir:
        results_directory = args.output_dir
    else:
        results_directory = "results/result_denoising_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    os.makedirs(results_directory, exist_ok=True)
    
    # Mostrar configuración cargada
    print("Configuración cargada:")
    print(f"- Arquitectura: {config['layers']}")
    print(f"- Learning rate: {config['learning_rate']}")
    print(f"- Función de activación: {config['function']}")
    print(f"- Optimizador: {config['optimizer']}")
    print(f"- Función de pérdida: {config['loss_function']}")
    print(f"- Épocas: {config['epochs']}")
    print(f"- Píxeles de ruido: {config['n_pixeles_ruido']}")
    print(f"- Directorio de resultados: {results_directory}")
    print("-" * 60)
    
    # Entrenar autoencoder
    entrenar_autoencoder(results_directory, config)
