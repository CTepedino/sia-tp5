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
    # Elegir la función de activación por nombre
    activador, activador_deriv = non_linear_functions["sigmoid"]

    # Guardar parámetros en JSON
    params = {
        "layers": [35, 12, 2, 12, 35],
        "learning_rate": 0.0043,
        "function": "sigmoid",
        "optimizer": "adam",
        "epochs": epochs
    }
    
    capa_latente = len(params["layers"]) // 2
    
    with open(os.path.join(results_directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    ae = MultiLayerPerceptron(
        layers=params["layers"],
        learning_rate=params["learning_rate"],
        activator_function=activador,
        activator_derivative=activador_deriv,
        optimizer=params["optimizer"]
    )

    ae.train(letras, letras, epochs=epochs)

    with open(os.path.join(results_directory, "result.txt"), "w") as f:
        errores_por_letra = []
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letra)
            error_letra = sum(abs(np.array(letra) - (np.array(reconstruida) > 0.5).astype(int)))
            errores_por_letra.append(error_letra)
            log_and_print(f"Letra {idx}: Error: {error_letra}", f)

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
        for letra in letras:
            _, activaciones = ae.forward_propagation(letra)
            z_list.append(activaciones[capa_latente])

        z = np.array(z_list)
        plt.scatter(z[:, 0], z[:, 1])
        for i in range(len(z)):
            plt.annotate(str(i), (z[i, 0], z[i, 1]))
        plt.title("Representación en el espacio latente (2D)")
        plt.grid(True)
        # Guardar el gráfico
        plt.savefig(os.path.join(results_directory, "espacio_latente.png"))
        plt.show()

if __name__ == "__main__":
    results_directory = "results/result_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(results_directory, exist_ok=True)
    entrenar_autoencoder(results_directory, epochs=50000)  # Reducimos el número de épocas
