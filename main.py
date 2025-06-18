import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron
import matplotlib.pyplot as plt
from activatorFunctions import non_linear_functions
import os
from datetime import datetime
import json
from collections import Counter


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

# Array de caracteres correspondiente a cada patrón de Font3
font3_chars = [
    "`", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "{", "|", "}", "~", "DEL"
]

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

def entrenar_autoencoder(results_directory, epochs=5000):
    letras = font_to_binary_patterns()
    activador, activador_deriv = non_linear_functions["sigmoid"]

    params = {
        # "layers": [35, 18, 6, 2, 6, 18, 35],
        "layers": [35, 20, 2, 20, 35],
        "learning_rate": 0.0024,
        "function": "sigmoid",
        "optimizer": "momentum",
        "epochs": epochs
    }
    
    with open(os.path.join(results_directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    ae = MultiLayerPerceptron(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        activator_function=activador,
        activator_derivative=activador_deriv,
        optimizer=params["optimizer"]
    )
    capa_latente = len(params["layers"]) // 2
    print(f"Capa latente: {capa_latente}")
    ae.train(letras, letras, epochs=epochs)

    with open(os.path.join(results_directory, "result.txt"), "w") as f:
        errores_por_letra = []
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letra)
            error_letra = sum(abs(np.array(letra) - (np.array(reconstruida) > 0.5).astype(int)))
            errores_por_letra.append(error_letra)
            log_and_print(f"Letra {font3_chars[idx]}: Error: {error_letra}", f)

        log_and_print(f"Error máximo por letra: {max(errores_por_letra):.6f}", f)
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
        for i, (error, count) in enumerate(zip(errors, counts)):
            plt.text(error, count + 0.1, str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, "distribucion_errores.png"))
        plt.show()

        # Visualizamos espacio latente
        print(f"Capa latente: {capa_latente}")
        z_list = []
        for letra in letras:
            _, activaciones = ae.forward_propagation(letra)
            z_list.append(activaciones[capa_latente])

        z = np.array(z_list)
        # Guardar los resultados de la capa latente en un archivo CSV (sin normalizar)
        with open(os.path.join(results_directory, "latentes.csv"), "w") as lat_file:
            lat_file.write("caracter,latente_x,latente_y\n")
            for i, vec in enumerate(z):
                lat_file.write(f"{font3_chars[i]},{vec[0]},{vec[1]}\n")
        # Normalizar a rango [0, 1] para mejor visualización
        z_min = z.min(axis=0)
        z_max = z.max(axis=0)
        z_norm = (z - z_min) / (z_max - z_min)
        # Guardar los resultados normalizados de la capa latente en un archivo CSV
        with open(os.path.join(results_directory, "latentes_normalizados.csv"), "w") as lat_file:
            lat_file.write("caracter,latente_x,latente_y\n")
            for i, vec in enumerate(z_norm):
                lat_file.write(f"{font3_chars[i]},{vec[0]},{vec[1]}\n")

        plt.figure(figsize=(10, 8))
        plt.scatter(z_norm[:, 0], z_norm[:, 1])
        for i in range(len(z_norm)):
            plt.annotate(font3_chars[i], (z_norm[i, 0], z_norm[i, 1]))
        plt.title("Representación en el espacio latente (2D)")
        plt.xlabel("Dimensión latente 1 (normalizada)")
        plt.ylabel("Dimensión latente 2 (normalizada)")
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-0.1, 1.1)
        plt.grid(True)
        plt.savefig(os.path.join(results_directory, "espacio_latente.png"))
        plt.show()

if __name__ == "__main__":
    results_directory = "results/result_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(results_directory, exist_ok=True)
    entrenar_autoencoder(results_directory, epochs=100000)  # Reducimos el número de épocas
