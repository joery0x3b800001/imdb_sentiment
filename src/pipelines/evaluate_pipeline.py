from src.config import Config
from src.data.loader import load_raw_splits
from src.data.preprocessing import tokenize_splits
from src.evaluation.evaluator import evaluate_model
from src.models.builder import build_tokenizer, load_trained_model

def run_evaluation(cfg: Config, model_dir: str):
    tokenizer = build_tokenizer(model_dir)
    raw = load_raw_splits(cfg.data, cfg.train.seed)
    data = tokenize_splits(raw, tokenizer, cfg.data)
    model = load_trained_model(model_dir, tokenizer)
    return evaluate_model(
        model, tokenizer, data["test"], model_dir,
        batch_size=cfg.train.eval_batch_size, tag="test",
    )