"""Evaluate any model on a tokenized dataset and persist a report."""
import json
import os

from sklearn.metrics import classification_report, confusion_matrix
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.constants import ID2LABEL
from src.training.metrics import compute_metrics
from src.utils.logging import get_logger

log = get_logger(__name__)

def evaluate_model(model, tokenizer, dataset, output_dir, batch_size=64, tag="test"):
    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    out = trainer.predict(dataset)
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    preds, labels = logits.argmax(-1), out.label_ids

    names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    report = {
        "metrics": {k.replace("test_", ""): v for k, v in out.metrics.items()},
        "classification_report": classification_report(
            labels, preds, target_names=names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("%s metrics: %s", tag, report["metrics"])
    log.info("Report saved to %s", path)
    return report