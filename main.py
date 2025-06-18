import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron
import matplotlib.pyplot as plt
from activatorFunctions import non_linear_functions
import os
from datetime import datetime


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


# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def contar_error_pixel(x, x_hat, umbral=0.5):
    errores = 0
    for orig, recon in zip(x, x_hat):
        errores += sum(abs(np.array(orig) - (np.array(recon) > umbral).astype(int)))
    return errores

def log_and_print(msg, file):
    print(msg)
    file.write(str(msg) + '\n')

def entrenar_autoencoder(results_directory):
    letras = font_to_binary_patterns()
    # Elegir la función de activación por nombre
    activador, activador_deriv = non_linear_functions["sigmoid"]  # o "sigmoid", "tanh", etc.

    ae = MultiLayerPerceptron(
        layers=[35, 10, 2, 10, 35],
        learning_rate=0.02,
        activator_function=activador,
        activator_derivative=activador_deriv,
        optimizer="adam"
    )

    ae.train(letras, letras, epochs=5000)

    with open(os.path.join(results_directory, "result.txt"), "w") as f:
        errores_por_letra = []
        for idx, letra in enumerate(letras):
            reconstruida = ae.test(letra)
            error_letra = sum(abs(np.array(letra) - (np.array(reconstruida) > 0.5).astype(int)))
            errores_por_letra.append(error_letra)
            log_and_print(f"Letra {idx}: Error: {error_letra}", f)

        log_and_print("Error máximo por letra: " + str(max(errores_por_letra)), f)
        log_and_print("Error promedio por letra: " + str(np.mean(errores_por_letra)), f)

        # Visualizamos espacio latente
        z_list = []
        for letra in letras:
            _, activaciones = ae.forward_propagation(letra)
            z_list.append(activaciones[3])  # capa latente

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
    entrenar_autoencoder(results_directory)
