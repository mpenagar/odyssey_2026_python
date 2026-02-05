import os
import requests

def download_data(name):
    path = os.path.join(os.path.dirname(__file__), name + '.npz')
    if not os.path.exists(path):
        BASE_URL = "https://github.com/mpenagar/odyssey_2026_python/releases/download/latest"
        URL = f"{BASE_URL}/{name}.npz"
        print(f"Descargando dataset {name} a {path} ...")
        response = requests.get(URL, stream=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Descarga completada.")
    else :
        print(f"Dataset {name} ya descargado en {path}.")
    return path
