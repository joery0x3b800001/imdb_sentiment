"""Lightweight inference wrapper."""
from typing import Dict, List

import torch

from src.constants import ID2LABEL
from src.data.preprocessing import clean_text
from src.models.builder import build_tokenizer, load_trained_model
from src.utils.device import get_device

class SentimentPredictor:
    def __init__(self, model_dir: str, max_length: int = 256, device: str = None):
        self.device = device or get_device()
        self.max_length = max_length
        self.tokenizer = build_tokenizer(model_dir)
        self.model = load_trained_model(model_dir, self.tokenizer).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = [clean_text(t) for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for text, p in zip(chunk, probs):
                idx = int(p.argmax())
                results.append({
                    "text": text[:80] + ("..." if len(text) > 80 else ""),
                    "label": ID2LABEL[idx],
                    "confidence": round(float(p[idx]), 4),
                })
        return results