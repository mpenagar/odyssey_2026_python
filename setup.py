# setup.py
from setuptools import setup, find_packages

setup(
    name="odyssey_2026",             # Nombre de importación
    version="0.1",
    packages=find_packages(),       # Busca automáticamente la carpeta 'odyssey_2026'
    install_requires=[              # Dependencias automáticas (Opcional)
        "numpy",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "requests"
    ]
)
