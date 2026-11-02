"""CLI entry point.

  python main.py train    --config configs/default.yaml
  python main.py evaluate --config configs/default.yaml --model_dir outputs/distilbert-imdb/final
  python main.py predict  --model_dir outputs/distilbert-imdb/final --text "Loved it!"
"""
import argparse

from src.config import load_config

def main():
    p = argparse.ArgumentParser(description="IMDB sentiment fine-tuning pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/default.yaml")

    e = sub.add_parser("evaluate")
    e.add_argument("--config", default="configs/default.yaml")
    e.add_argument("--model_dir", required=True)

    r = sub.add_parser("predict")
    r.add_argument("--model_dir", required=True)
    r.add_argument("--text", nargs="+", help="One or more reviews")
    r.add_argument("--file", help="Text file with one review per line")
    r.add_argument("--max_length", type=int, default=256)

    a = p.parse_args()

    if a.cmd == "train":
        from src.pipelines.train_pipeline import run_training
        run_training(load_config(a.config))
    elif a.cmd == "evaluate":
        from src.pipelines.evaluate_pipeline import run_evaluation
        run_evaluation(load_config(a.config), a.model_dir)
    else:
        from src.pipelines.predict_pipeline import run_prediction
        texts = a.text or []
        if a.file:
            with open(a.file) as f:
                texts += [l.strip() for l in f if l.strip()]
        if not texts:
            p.error("provide --text or --file")
        for res in run_prediction(a.model_dir, texts, a.max_length):
            print(f"{res['label']:>8} ({res['confidence']:.2%})  {res['text']}")

if __name__ == "__main__":
    main()