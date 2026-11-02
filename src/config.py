"""Typed configuration loaded from YAML."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml

@dataclass
class DataConfig:
    dataset_name: str = "stanfordnlp/imdb"
    max_length: int = 256
    val_size: float = 0.1
    train_subset: Optional[int] = None
    test_subset: Optional[int] = None
    num_proc: int = 4

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    grad_accum_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2   # 0 disables early stopping
    logging_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    report_to: str = "none"            # "wandb" / "tensorboard" if desired

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )