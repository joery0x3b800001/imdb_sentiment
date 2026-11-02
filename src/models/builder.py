"""Model / tokenizer construction and loading of trained checkpoints."""
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ModelConfig
from src.constants import ID2LABEL, LABEL2ID

def build_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:                 # decoder LLMs (Qwen, Llama, GPT-2...)
        tok.pad_token = tok.eos_token
    return tok

def build_model(cfg: ModelConfig, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,       # also keeps the classifier head trainable
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    return model

def load_trained_model(model_dir: str, tokenizer=None):
    """Load a saved full model or a LoRA adapter directory."""
    is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    kwargs = dict(num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)
    if is_adapter:
        from peft import AutoPeftModelForSequenceClassification

        model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    if tokenizer is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model