"""Research-grade PyTorch and PEFT LoRA training module for DeBERTa-v3 NLI."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.metrics import compute_all_metrics
from src.utils.checkpoints import ModelCheckpointTracker

logger = logging.getLogger(__name__)


class NLIPairDataset(Dataset):
    """PyTorch Dataset for Premise-Hypothesis NLI training."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        premise = str(item.get("premise", ""))
        hypothesis = str(item.get("hypothesis", ""))
        label = int(item.get("label", 0))  # 1: entailment/support, 0: contradiction/hallucinated

        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_utilization_lora(
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    output_dir: str | Path,
    model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train DeBERTa-v3-base MNLI using PEFT LoRA.
    Evaluates on validation set after each epoch.
    Strictly follows research checkpoint rule:
    Saves:
    - checkpoint-last
    - checkpoint-best-auprc (best validation AUPRC)
    - checkpoint-best-ece (best validation ECE)
    Never selects checkpoint using test set!
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)

    # Attach PEFT LoRA
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_proj", "value_proj"]
        )
        model = get_peft_model(base_model, peft_config)
        logger.info("Initialized PEFT LoRA adapter for DeBERTa-v3.")
    except ImportError:
        logger.warning("peft not available, training top classification head.")
        model = base_model

    model.to(device)

    train_dataset = NLIPairDataset(train_pairs, tokenizer)
    val_dataset = NLIPairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    tracker = ModelCheckpointTracker(checkpoints_dir=out_path)

    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Evaluate strictly on validation split
        model.eval()
        val_probs_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                # Probability of entailment vs contradiction/neutral
                probs = torch.softmax(logits, dim=-1)[:, -1].cpu().numpy()
                val_probs_list.extend(probs)
                val_labels_list.extend(labels.numpy())

        val_metrics = compute_all_metrics(
            y_true=np.array(val_labels_list),
            y_prob=np.array(val_probs_list),
            threshold=0.5
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics
        }
        training_log.append(epoch_record)
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val AUPRC: {val_metrics['auprc']:.4f} | Val ECE: {val_metrics['ece']:.4f}")

        # Save checkpoint callback
        def save_checkpoint(ckpt_dir: Path):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir)
            else:
                torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(ckpt_dir)

        tracker.update(epoch_or_step=epoch, metrics=val_metrics, save_fn=save_checkpoint)

    # Save training history
    with open(out_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    return {
        "best_auprc": tracker.best_auprc,
        "best_ece": tracker.best_ece,
        "history": training_log
    }
