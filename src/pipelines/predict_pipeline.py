from typing import List

from src.inference.predictor import SentimentPredictor

def run_prediction(model_dir: str, texts: List[str], max_length: int = 256):
    return SentimentPredictor(model_dir, max_length=max_length).predict(texts)