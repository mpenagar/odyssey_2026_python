import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, det_curve
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np

def target_scores(y, score_matrix):
    """
    Extrae el vector de scores y targets (1/0) a partir del vector de etiquetas
    y la matriz cuadrada que contiene los scores de todos los cruces.

    y: Vector de etiquetas (N_samples,)
    score_matrix: matriz de scores (N_samples,N_samples)
    """

    # 1. Crear Matriz de Ground Truth (N x N)
    # True si son el mismo locutor, False si son diferentes
    # Truco de broadcasting: Compara columna contra fila
    ground_truth_matrix = (y[:, np.newaxis] == y[np.newaxis, :])

    # 2. Extraer solo el triángulo superior (sin diagonal)
    # Esto evita duplicados (A vs B es lo mismo que B vs A) y auto-comparación (A vs A)
    mask = np.triu(np.ones_like(ground_truth_matrix, dtype=bool), k=1)

    scores = score_matrix[mask]       # Lista plana de similitudes
    target = ground_truth_matrix[mask] # Lista plana de True/False

    # Convertir etiquetas a enteros (1 = Mismo Locutor, 0 = Diferente)
    target = target.astype(int)

    return scores, target

def eer(scores, target):
    """
    Estima el Equal Error Rate

    scores: vector de scores
    target: vector de targets (1:target, 0:non-target)
    """
    
    print(f"Total de scores evaluados: {len(scores)}")
    print(f"   - Scores Positivos (target): {np.sum(target)}")
    print(f"   - Scores Negativos (non-target): {len(target) - np.sum(target)}")

    # 1. Calcular Curva ROC y EER
    fpr, tpr, thresholds = roc_curve(target, scores)
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

    return eer, threshold_eer


def plot_hist(scores, target, threshold_eer, xlabel='score'):
    """
    Muestra el histograma de scores

    scores: vector de scores
    target: vector de targets (1:target, 0:non-target)
    threshold_eer: umbral de EER
    """
    
    # Separar puntuaciones
    pos_scores = scores[target == 1] # Mismo locutor
    neg_scores = scores[target == 0] # Diferente locutor

    plt.figure(figsize=(10, 5))

    # Histograma
    plt.hist(pos_scores, bins=50, alpha=0.5, color='green', label='Target', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.5, color='red', label='Impostor', density=True)

    # Línea del umbral
    plt.axvline(threshold_eer, color='black', linestyle='--', label=f'Umbral EER ({threshold_eer:.2f})')

    plt.xlabel(xlabel)
    plt.ylabel('Densidad')
    plt.title('Distribución de Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_det_curve(scores, target, label='Sistema'):
    """
    Genera una curva DET usando la escala de Desviación Normal (Probit),
    haciendo que las distribuciones gaussianas se vean como líneas rectas.

    scores: vector de scores
    target: vector de targets (1:target, 0:non-target)
    threshold_eer: umbral de EER
    """    """
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

def train_test_y_split(X, y, test_size=0.5, random_state=42):
    """
    Separa train/test por etiquetas y. El conjunto resultante de
    etiquetas de X_train y X_test es disjunto
    """

    # Obtener las IDs únicas
    y_unique = np.unique(y)

    # Dividir las IDs únicas en entrenamiento y test
    train_ids, test_ids = train_test_split(y_unique, test_size=test_size, random_state=random_state)

    # Crear máscaras para seleccionar los vectores y etiquetas correspondientes
    train_mask = np.isin(y, train_ids)
    test_mask  = np.isin(y, test_ids)

    # Aplicar las máscaras para obtener los conjuntos de entrenamiento y prueba
    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Shape de X_train: {X_train.shape}")
    print(f"Shape de y_train: {y_train.shape}")
    print(f"Número de speakers únicos en train: {len(np.unique(y_train))}")

    print(f"\nShape de X_test: {X_test.shape}")
    print(f"Shape de y_test: {y_test.shape}")
    print(f"Número de speakers únicos en test: {len(np.unique(y_test))}")

    # Verificar que no haya solapamiento de speakers
    overlap_speakers = np.intersect1d(np.unique(y_train), np.unique(y_test))
    print(f"\nNúmero de speakers solapados entre train y test: {len(overlap_speakers)}")

    return X_train, X_test, y_train, y_test
