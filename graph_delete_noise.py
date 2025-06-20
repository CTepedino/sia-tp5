import matplotlib.pyplot as plt
import numpy as np

# Datos de entrada
pixeles_modificados = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
promedio_error_por_letra = [0, 0, 0.0625, 0.0625, 0.0625, 0.1875, 0.28125, 0.28125, 0.15625, 0.375, 0.09375, 0.09375, 0.25, 0.1875, 0.15625, 0.0625, 0, 0]

# Gráfico 1: Línea de tendencia
plt.figure(figsize=(10, 6))
plt.plot(pixeles_modificados, promedio_error_por_letra, 'bo-', linewidth=2, markersize=8, label='Error promedio')
plt.xlabel('Número de píxeles modificados', fontsize=12)
plt.ylabel('Error promedio por letra', fontsize=12)
plt.title('Error de Reconstrucción vs Píxeles Modificados', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Agregar etiquetas de valor en los puntos
for i, (x, y) in enumerate(zip(pixeles_modificados, promedio_error_por_letra)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=9, fontweight='bold')

# Agregar línea de tendencia polinomial
if len(pixeles_modificados) > 3:
    z = np.polyfit(pixeles_modificados, promedio_error_por_letra, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(pixeles_modificados), max(pixeles_modificados), 100)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7, linewidth=2, label='Tendencia polinomial')
    plt.legend()

plt.tight_layout()
plt.savefig('error_vs_pixeles_linea.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 2: Barras
plt.figure(figsize=(10, 6))
bars = plt.bar(pixeles_modificados, promedio_error_por_letra, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
plt.xlabel('Número de píxeles modificados', fontsize=12)
plt.ylabel('Error promedio por letra', fontsize=12)
plt.title('Error de Reconstrucción vs Píxeles Modificados (Barras)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Agregar etiquetas de valor en las barras
for bar, error in zip(bars, promedio_error_por_letra):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{error:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('error_vs_pixeles_barras.png', dpi=300, bbox_inches='tight')
plt.show()








