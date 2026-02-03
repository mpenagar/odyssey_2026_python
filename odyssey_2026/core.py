# odyssey_2026/core.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, det_curve
from scipy.stats import norm

def evaluate(y, score_matrix):
    """
    y: Vector de etiquetas (N_samples,)
    score_matrix: matriz de scores (N_samples,N_samples)
    """

    print(f"--- Iniciando Benchmark con {len(y)} muestras ---")

    # 2. Crear Matriz de Ground Truth (N x N)
    # True si son el mismo locutor, False si son diferentes
    # Truco de broadcasting: Compara columna contra fila
    ground_truth_matrix = (y[:, np.newaxis] == y[np.newaxis, :])

    # 3. Extraer solo el triángulo superior (sin diagonal)
    # Esto evita duplicados (A vs B es lo mismo que B vs A) y auto-comparación (A vs A)
    mask = np.triu(np.ones_like(ground_truth_matrix, dtype=bool), k=1)

    scores = score_matrix[mask]       # Lista plana de similitudes
    labels = ground_truth_matrix[mask] # Lista plana de True/False

    # Convertir etiquetas a enteros (1 = Mismo Locutor, 0 = Diferente)
    labels = labels.astype(int)

    print(f"Total de pares evaluados: {len(scores)}")
    print(f"   - Pares Positivos (Mismo Locutor): {np.sum(labels)}")
    print(f"   - Pares Negativos (Diferente Locutor): {len(labels) - np.sum(labels)}")

    # 4. Calcular Curva ROC y EER
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # El EER es el punto donde FPR se cruza con FNR
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx_eer]
    threshold_eer = thresholds[idx_eer]

    print(f"\nResultados:")
    print(f"---------------------------------------")
    print(f"EER (Equal Error Rate): {eer*100:.2f}%")
    print(f"Umbral óptimo (Similitud): {threshold_eer:.4f}")
    print(f"---------------------------------------")

    return scores, labels, eer, threshold_eer

def plot_results(scores, labels, threshold, xlabel='score'):
    # Separar puntuaciones
    pos_scores = scores[labels == 1] # Mismo locutor
    neg_scores = scores[labels == 0] # Diferente locutor

    plt.figure(figsize=(10, 5))

    # Histograma
    plt.hist(pos_scores, bins=50, alpha=0.5, color='green', label='Target', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.5, color='red', label='Impostor', density=True)

    # Línea del umbral
    plt.axvline(threshold, color='black', linestyle='--', label=f'Umbral EER ({threshold:.2f})')

    plt.xlabel(xlabel)
    plt.ylabel('Densidad')
    plt.title('Distribución de Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_det_curve(labels, scores, label='Sistema'):
    """
    Genera una curva DET usando la escala de Desviación Normal (Probit),
    haciendo que las distribuciones gaussianas se vean como líneas rectas.
    """
    # 1. Calcular FPR y FNR
    fpr, fnr, _ = det_curve(labels, scores, pos_label=1)

    # 2. Transformación a Escala Normal (Probit)
    # ppf es la 'Percent Point Function' (inversa de la CDF)
    # Evitamos 0 y 1 absolutos para que no den infinito
    fpr_probit = norm.ppf(np.clip(fpr, 1e-6, 1 - 1e-6))
    fnr_probit = norm.ppf(np.clip(fnr, 1e-6, 1 - 1e-6))

    plt.figure(figsize=(8, 8))

    # 3. Plotear en el espacio transformado
    plt.plot(fpr_probit, fnr_probit, linewidth=2, label=label)

    # 4. Configurar los Ejes (La parte mágica)
    # Definimos los ticks que queremos ver en % (ej. 0.1%, 1%, 50%...)
    # Pero los colocamos en su posición 'probit'
    ticks_percent = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 50, 70, 90])
    ticks_probit = norm.ppf(ticks_percent / 100) # Convertir % a probabilidad 0-1 y luego a probit

    plt.xticks(ticks_probit, labels=[str(t) for t in ticks_percent])
    plt.yticks(ticks_probit, labels=[str(t) for t in ticks_percent])

    # 5. Límites y Estética
    # Ajustamos la ventana para ver la zona de interés (ej. 0.05% a 50%)
    limit_min = norm.ppf(0.0005) # 0.05%
    limit_max = norm.ppf(0.50)   # 50%

    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)

    # Línea de EER (Diagonal en este espacio)
    plt.plot([limit_min, limit_max], [limit_min, limit_max],
             linestyle='--', color='gray', alpha=0.5, label='EER Reference')

    plt.xlabel('False Alarm (%)')
    plt.ylabel('Miss (%)')
    plt.title('Curva DET')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.show()
