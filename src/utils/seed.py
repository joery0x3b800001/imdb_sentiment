from transformers import set_seed as _set_seed

def set_seed(seed: int = 42):
    _set_seed(seed)
