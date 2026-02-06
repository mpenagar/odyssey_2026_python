import torch
import numpy as np
import random
import os

def seed(seed=42):
    """
    Fija la semilla en todos los generadores de números aleatorios
    para garantizar resultados deterministas (reproducibles).
    """
    # 1. Python nativo
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Numpy
    np.random.seed(seed)

    # 3. PyTorch (CPU y GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Para multi-GPU

    # 4. Backend de CuDNN (La parte "oculta")
    # A veces, para ir más rápido, CUDA elige algoritmos no deterministas.
    # Esto fuerza a usar siempre los mismos, aunque sea un poco más lento.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
