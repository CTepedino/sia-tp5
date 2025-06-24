
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

# Autoencoder Variacional para Generación de Emojis

Este proyecto implementa un **autoencoder variacional (VAE)** que entrena con un conjunto de emojis y permite visualizar el espacio latente, reconstruir los emojis originales y generar nuevos mediante interpolación entre embeddings.

---

## Configuración

El sistema se configura desde un archivo JSON. Ejemplo de configuración:

```json
{
    "latent_size": 2,
    "hidden_sizes": [100],
    "learning_rate": 0.001,
    "epochs": 15000,
    "activator": "sigmoid",
    "output_activator": "sigmoid",
    "emoji_indexes": [0, 1, 2, 3],
    "num_interpolations": 5,
    "interpolation_steps": 10
}
```

### Parámetros:

- **latent_size**: Dimensión del espacio latente (debe ser 2 para visualización 2D)
- **hidden_sizes**: Lista de tamaños de capas ocultas simétricas
- **learning_rate**: Tasa de aprendizaje para todas las capas
- **epochs**: Número de épocas de entrenamiento
- **activator**: Función de activación para capas ocultas (`sigmoid`, `tanh`, `relu`)
- **output_activator**: Activación de la capa final
- **emoji_indexes** *(opcional)*: Índices de los emojis del conjunto a usar
- **num_interpolations** *(opcional)*: Cantidad de interpolaciones a generar entre pares aleatorios
- **interpolation_steps** *(opcional)*: Número de pasos intermedios entre dos emojis

---

## Uso

### Entrenamiento y generación:
```bash
uv run ./main_vae.py config.json
```

Este comando:
1. Entrena el modelo VAE con los emojis seleccionados
2. Guarda reconstrucciones y visualizaciones
3. Genera emojis nuevos mediante interpolación

---

## Resultados

Los resultados se guardan automáticamente en un directorio `results/YYYY-MM-DD_HH-MM-SS`, e incluyen:

- `config.json`: Configuración usada
- `vae_weights.npz`: Pesos entrenados
- `emoji_grid.png`: Emojis originales usados para entrenamiento
- `vae_recon_grid.png`: Reconstrucciones desde el autoencoder
- `reconstruction_*.png`: Comparación visual original vs reconstrucción
- `interpolation_*.png`: Interpolaciones lineales entre emojis
- `latent_space.png`: Espacio latente si `latent_size = 2`
- `reconstruction_mse_histogram.png`: Histograma de error de reconstrucción

---

## Funcionalidad

- **Reconstrucción de emojis**: Evalúa la fidelidad del autoencoder
- **Visualización del espacio latente**: Si `latent_size == 2`
- **Interpolación**: Generación de nuevos emojis interpolando entre embeddings
- **Persistencia**: Guarda pesos y configuración automáticamente

---

## Notas

- El conjunto de emojis se carga desde un sprite PNG (`emojis.png`) precortado en bloques de 20×20 píxeles.
- El modelo puede cargarse desde pesos previamente entrenados para evitar retraining.
- Se puede modificar fácilmente para otros datasets de imágenes binarias (e.g., letras o dígitos).
