from ._utils import download_data
from pathlib import Path
import numpy as np

_path = download_data(Path(__file__).stem)
_data = np.load(_path, allow_pickle=True)

embeddings = _data['embeddings']
spk_ids =    _data['speakers']
