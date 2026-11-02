"""End-to-end: data -> tokenize -> model -> train -> test evaluation -> save."""
import json
import os

from src.config import Config
from src.data.loader import load_raw_splits
from src.data.preprocessing import tokenize_splits
from src.evaluation.evaluator import evaluate_model
from src.models.builder import build_model, build_tokenizer
from src.training.trainer import build_trainer
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

def run_training(cfg: Config) -> str:
    set_seed(cfg.train.seed)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    with open(os.path.join(cfg.train.output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    tokenizer = build_tokenizer(cfg.model.model_name)
    raw = load_raw_splits(cfg.data, cfg.train.seed)
    data = tokenize_splits(raw, tokenizer, cfg.data)
    model = build_model(cfg.model, tokenizer)

    trainer = build_trainer(model, tokenizer, data["train"], data["validation"], cfg.train)
    trainer.train()

    final_dir = os.path.join(cfg.train.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("Model saved to %s", final_dir)

    evaluate_model(
        trainer.model, tokenizer, data["test"], final_dir,
        batch_size=cfg.train.eval_batch_size, tag="test",
    )
    return final_dir