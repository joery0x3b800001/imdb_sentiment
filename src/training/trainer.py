"""Builds the HF Trainer from config."""
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import TrainConfig
from src.training.metrics import compute_metrics
from src.utils.device import get_accelerator, supports_bf16

def build_trainer(model, tokenizer, train_ds, val_ds, cfg: TrainConfig) -> Trainer:
    accelerator = get_accelerator()
    bf16 = accelerator is not None and supports_bf16(accelerator)
    fp16 = accelerator is not None and accelerator.type in {"cuda", "xpu"} and not bf16

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=cfg.logging_steps,
        bf16=bf16,
        fp16=fp16,
        seed=cfg.seed,
        report_to=cfg.report_to,
    )
    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(cfg.early_stopping_patience))

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )