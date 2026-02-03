# odyssey_2026/data/librispeech-pyannote_embeddings.py
import os
import numpy as np

# 1. Ruta relativa a ESTE archivo
_path = os.path.join(os.path.dirname(__file__), 'librispeech_pyannote.npz')

# 2. Carga inmediata al importar este módulo específico
print("Cargando LibriSpeech...")
_data = np.load(_path)

embeddings = _data['X']
spk_ids = _data['y']
