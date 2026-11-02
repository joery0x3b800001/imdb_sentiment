# IMDB Sentiment Fine-Tuning Pipeline

Fine-tunes a small language model on the IMDB reviews dataset
(25k train / 25k test, balanced) for binary sentiment classification.

## Layout
```
main.py                     CLI: train | evaluate | predict
configs/                    YAML configs (default = DistilBERT full FT, lora_qwen = Qwen2.5-0.5B + LoRA)
src/
  config.py                 typed dataclass config
  constants.py              label maps
  data/loader.py            load + stratified train/val/test split
  data/preprocessing.py     HTML cleanup + tokenization
  models/builder.py         tokenizer/model (optional LoRA), checkpoint loading
  training/trainer.py       HF Trainer setup (early stopping, bf16/fp16 auto)
  training/metrics.py       accuracy / precision / recall / F1
  evaluation/evaluator.py   test-set eval + JSON report + confusion matrix
  inference/predictor.py    batch inference class
  pipelines/                train / evaluate / predict orchestration
  utils/                    seeding, logging
```

## Usage
```bash
pip install -r requirements.txt

# quick smoke test: set data.train_subset: 2000 and data.test_subset: 1000 in the config
python main.py train --config configs/default.yaml

# LoRA on a small decoder LLM
python main.py train --config configs/lora_qwen.yaml

python main.py evaluate --config configs/default.yaml --model_dir outputs/distilbert-imdb/final
python main.py predict  --model_dir outputs/distilbert-imdb/final \
    --text "An absolute masterpiece." "Two hours of my life I will never get back."
```

## Notes
- Any `AutoModelForSequenceClassification`-compatible model works: change `model.model_name`.
- Inference uses an available PyTorch accelerator (such as CUDA, Apple MPS, or Intel XPU), falling back to CPU.
- Padding is dynamic per batch; decoder tokenizers get `pad_token = eos_token` automatically.
- Typical results: DistilBERT ~92-93% test accuracy; larger models/LoRA slightly higher.
- Want more data? Point `data.dataset_name` at another dataset with `text`/`label` columns
  (e.g. `rotten_tomatoes`, `yelp_polarity`) or concatenate datasets in `data/loader.py`.
