# odyssey_2026/data/librispeech_pyannote.py
import os
import numpy as np
import requests

_path = os.path.join(os.path.dirname(__file__), 'librispeech_pyannote.npz')
_URL_DESCARGA = "https://github.com/mpenagar/odyssey_2026_python/releases/download/latest/librispeech_pyannote.npz"

def _download_data():
    print(f"Descargando dataset a {_path} ...")
    response = requests.get(_URL_DESCARGA, stream=True)
    with open(_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Descarga completada.")

_download_data()
_data = np.load(_path, allow_pickle=True)

embeddings = _data['embeddings']
spk_ids =    _data['speakers']
