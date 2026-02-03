# setup.py
from setuptools import setup, find_packages

setup(
    name="odyssey_2026",             # Nombre de importación
    version="0.1",
    packages=find_packages(),       # Busca automáticamente la carpeta 'odyssey_2026'
    package_data={
        'odyssey_2026': ['data/*.npz']  # Busca dentro de la carpeta data
    },
    install_requires=[              # Dependencias automáticas (Opcional)
        "numpy",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "requests"
    ]
)
