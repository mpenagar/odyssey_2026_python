import numpy as np
from ._utils import download_data

# Generación dinámica de módulos odyssey_2026.data.xxx que hacen referencia
# a datasets guardados en ficheros xxx.npz alojados en la release.

# 1. Definimos qué datasets existen realmente
# Esto es importante para que el sistema sepa qué descargar
_AVAILABLE_DATASETS = [
    'librispeech_dev_clean_pyannote_restnet34', 
    'librispeech_train_clean_100_pyannote_restnet34', 
    'librispeech_train_clean_360_pyannote_restnet34'
]

# 2. Esta función se ejecuta cuando alguien importa algo que NO existe como fichero
def __getattr__(name):
    
    # Si piden algo que no está en nuestra lista, error normal
    if name not in _AVAILABLE_DATASETS:
        raise AttributeError(f"El dataset '{name}' no existe en odyssey_2026.data")
    
    # --- LOGICA DE CARGA AUTOMÁTICA ---
    print(f"Cargando dataset dinámico: {name} ...")
    
    # Usamos el 'name' (ej: 'librispeech_pyannote') para descargar/buscar
    path = download_data(name) 
    
    # Cargamos el npz
    data = np.load(path, allow_pickle=True)
    
    # 3. Crear un objeto "falso" que actúe como módulo
    # Usamos una clase simple para guardar los datos
    class DatasetContainer:
        def __repr__(self):
            return f"<Dataset {name}: {data['embeddings'].shape}>"
            
    dataset = DatasetContainer()
    
    # Asignamos los atributos que espera el usuario
    dataset.embeddings = data['embeddings']
    dataset.spk_ids = data['speakers']
    
    # Alias X e y (por si acaso)
    dataset.X = dataset.embeddings
    dataset.y = dataset.spk_ids
    
    return dataset

# 3. Esto ayuda al autocompletado del IDE (VS Code / Colab)
# Para que sepa qué opciones sugerir al escribir "voice_tools.data."
def __dir__():
    return __all__ + _AVAILABLE_DATASETS
