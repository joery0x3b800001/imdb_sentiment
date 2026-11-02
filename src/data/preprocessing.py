"""Text cleaning + tokenization."""
import re

from datasets import DatasetDict

from src.config import DataConfig

_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = _BR.sub(" ", text)          # IMDB reviews contain <br /> tags
    return _WS.sub(" ", text).strip()

def tokenize_splits(splits: DatasetDict, tokenizer, cfg: DataConfig) -> DatasetDict:
    """Tokenize without padding; padding is done dynamically per batch."""

    def _fn(batch):
        return tokenizer(
            [clean_text(t) for t in batch["text"]],
            truncation=True,
            max_length=cfg.max_length,
        )

    out = splits.map(
        _fn,
        batched=True,
        num_proc=cfg.num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return out.rename_column("label", "labels")