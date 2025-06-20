# Autoencoder para Generación de Letras

Este proyecto implementa un autoencoder que puede generar nuevas letras interpolando en el espacio latente entre letras existentes del conjunto de entrenamiento.

## Configuración

El sistema utiliza un archivo JSON de configuración para todos los parámetros. Ejemplo de configuración:

```json
{
    "layers": [35, 17, 2, 17, 35],
    "learning_rate": 0.009,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 5000,
    "interpolation": {
        "letra1_idx": 1,
        "letra2_idx": 15,
        "n_interpolations": 5
    }
}
```

### Parámetros de configuración:

- **layers**: Arquitectura de la red neuronal (debe ser simétrica para autoencoder)
- **learning_rate**: Tasa de aprendizaje
- **function**: Función de activación ("sigmoid", "tanh", "relu")
- **optimizer**: Optimizador ("adam", "momentum", "gradient")
- **loss_function**: Función de pérdida ("mse", "binary_crossentropy")
- **epochs**: Número de épocas de entrenamiento
- **interpolation**:
  - **letra1_idx**: Índice de la primera letra para interpolación (0-31)
  - **letra2_idx**: Índice de la segunda letra para interpolación (0-31)
  - **n_interpolations**: Número de letras intermedias a generar

### Mapeo de letras (índices 0-31):
```
0: `, 1: a, 2: b, 3: c, 4: d, 5: e, 6: f, 7: g,
8: h, 9: i, 10: j, 11: k, 12: l, 13: m, 14: n, 15: o,
16: p, 17: q, 18: r, 19: s, 20: t, 21: u, 22: v, 23: w,
24: x, 25: y, 26: z, 27: {, 28: |, 29: }, 30: ~, 31: DEL
```

## Uso

### Ejecución básica:
```bash
python main_autoencoder.py config_autoencoder.json
```

### Con directorio de salida personalizado:
```bash
uv run ./main_autoencoder.py config_autoencoder.json --output-dir results/mi_experimento
```

### Ver ayuda:
```bash
uv run ./main_autoencoder.py --help
```

## Funcionalidad

El sistema realiza las siguientes operaciones:

1. **Entrenamiento del autoencoder** con las letras del conjunto Font3
2. **Reconstrucción de letras** para evaluar el rendimiento
3. **Visualización del espacio latente** en 2D
4. **Generación de nuevas letras** mediante interpolación lineal en el espacio latente
5. **Guardado de resultados** en archivos PNG y CSV

## Resultados

Los resultados se guardan en el directorio especificado:

- `params.json`: Parámetros utilizados
- `result.txt`: Errores de reconstrucción por letra
- `distribucion_errores.png`: Gráfico de distribución de errores
- `espacio_latente.png`: Visualización del espacio latente
- `interpolacion_lineal.png`: Letras generadas por interpolación
- `letras_interpoladas.csv`: Datos de las letras interpoladas
- `resultado_letras.csv`: Reconstrucciones de todas las letras
- `letter_map.png`: Mapa visual de todas las letras reconstruidas

## Ejemplos de configuración

```json
{
    "layers": [35, 17, 2, 17, 35],
    "learning_rate": 0.009,
    "function": "sigmoid",
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "epochs": 5000,
    "interpolation": {
        "letra1_idx": 1,
        "letra2_idx": 15,
        "n_interpolations": 3
    }
}
```

## Notas

- Los pesos entrenados se guardan automáticamente en la carpeta `weights/`
- Si se encuentran pesos previos, el entrenamiento continúa desde ese punto
- El sistema valida automáticamente los índices de letras en la configuración
- Las letras generadas son completamente nuevas y no pertenecen al conjunto de entrenamiento 