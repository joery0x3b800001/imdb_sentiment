"""Raw dataset loading and splitting (no tokenization here)."""
from datasets import DatasetDict, load_dataset

from src.config import DataConfig
from src.utils.logging import get_logger

log = get_logger(__name__)

def _subset(ds, n, seed):
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))

def load_raw_splits(cfg: DataConfig, seed: int = 42) -> DatasetDict:
    """Return DatasetDict with train / validation / test.

    IMDB ships 25k train + 25k test (balanced). We carve a stratified
    validation set out of train so the test set stays untouched.
    """
    ds = load_dataset(cfg.dataset_name)
    split = ds["train"].train_test_split(
        test_size=cfg.val_size, seed=seed, stratify_by_column="label"
    )
    splits = DatasetDict(
        train=_subset(split["train"], cfg.train_subset, seed),
        validation=split["test"],
        test=_subset(ds["test"], cfg.test_subset, seed),
    )
    log.info({k: len(v) for k, v in splits.items()})
    return splits