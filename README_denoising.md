
# TP5 - SIA - Grupo 1

## Requisitos

- Python 3.8 o superior
- uv: gestor de entornos y dependencias

### Instalación de uv:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### Sincronización de dependencias:

Este proyecto incluye un archivo requirements.txt. Para instalar todo con uv, ejecutar:

```bash
uv venv .venv  
uv sync
```

Esto crea el entorno virtual en .venv e instala automáticamente las dependencias necesarias.

---

# Autoencoder de Denoising para Letras

Este proyecto implementa un autoencoder de denoising que intenta limpiar letras ruidosas y reconstruir las letras originales del conjunto Font3.

## Configuración

El sistema utiliza un archivo JSON de configuración para todos los parámetros. Ejemplo de configuración:

```json
{
    "layers": [35, 17, 2, 17, 35],
    "learning_rate": 0.007,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 5000,
    "n_pixeles_ruido": 3
}
```

### Parámetros de configuración:

- **layers**: Arquitectura de la red neuronal (debe ser simétrica para autoencoder)
- **learning_rate**: Tasa de aprendizaje
- **function**: Función de activación ("sigmoid", "tanh", "relu")
- **optimizer**: Optimizador ("adam", "momentum", "gradient")
- **loss_function**: Función de pérdida ("mse", "binary_crossentropy")
- **epochs**: Número de épocas de entrenamiento
- **n_pixeles_ruido**: Número de píxeles que se modifican aleatoriamente para crear ruido

## Uso

### Ejecución básica:
```bash
uv run ./main_denoising.py config_denoising.json
```

### Con directorio de salida personalizado:
```bash
uv run ./main_denoising.py config_denoising.json --output-dir results/mi_experimento_denoising
```

### Ver ayuda:
```bash
uv run ./main_denoising.py --help
```

## Funcionalidad

El sistema realiza las siguientes operaciones:

1. **Generación de ruido**: Modifica aleatoriamente un número específico de píxeles en cada letra
2. **Entrenamiento del autoencoder** con letras ruidosas como entrada y letras originales como objetivo
3. **Reconstrucción de letras ruidosas** para evaluar el rendimiento de limpieza
4. **Visualización del espacio latente** en 2D
5. **Guardado de resultados** en archivos PNG y CSV

## Resultados

Los resultados se guardan en el directorio especificado:

- `params.json`: Parámetros utilizados
- `result.txt`: Errores de reconstrucción por letra
- `distribucion_errores.png`: Gráfico de distribución de errores
- `espacio_latente.png`: Visualización del espacio latente
- `resultado_letras.csv`: Reconstrucciones de todas las letras
- `letter_map.png`: Mapa visual de todas las letras reconstruidas (limpias)
- `letter_map_ruido.png`: Mapa visual de todas las letras de entrada (con ruido)

## Ejemplos de configuración

### Configuración con poco ruido:
```json
{
    "layers": [35, 17, 2, 17, 35],
    "learning_rate": 0.007,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 5000,
    "n_pixeles_ruido": 1
}
```

### Configuración con mucho ruido:
```json
{
    "layers": [35, 20, 2, 20, 35],
    "learning_rate": 0.005,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 8000,
    "n_pixeles_ruido": 10
}
```

### Configuración rápida para pruebas:
```json
{
    "layers": [35, 17, 2, 17, 35],
    "learning_rate": 0.01,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 1000,
    "n_pixeles_ruido": 3
}
```

## Diferencias con el Autoencoder Estándar

- **Entrada**: Letras con ruido (píxeles modificados aleatoriamente)
- **Objetivo**: Letras originales sin ruido
- **Propósito**: Aprender a limpiar ruido y reconstruir la información original

## Notas

- Los pesos entrenados se guardan automáticamente en la carpeta `weights/` con sufijo `_denoising`
- Si se encuentran pesos previos, el entrenamiento continúa desde ese punto
- El ruido se genera aleatoriamente en cada ejecución
- El sistema muestra tanto las letras de entrada (con ruido) como las de salida (limpias)
- El error se mide comparando las letras reconstruidas con las letras originales (sin ruido) 