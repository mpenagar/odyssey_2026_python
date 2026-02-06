# odyssey_2026/__init__.py

# Importamos las funciones de core.py para que sean accesibles directamente
from .eval import target_scores, get_eer, plot_hist, plot_det_curve, train_test_y_split
from .deterministic import seed
