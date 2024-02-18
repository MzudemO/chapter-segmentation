import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForNextSentencePrediction,
    BertTokenizer,
    get_scheduler,
    TrainingArguments,
    Trainer,
)
import transformers
import evaluate
import json
import numpy as np

from utils import preprocess

metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(predictions)
    print(labels)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # Model and device setup
    if torch.cuda.device_count() > 0:
        print(
            f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
        )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hyperparams = {
        "model": "deepset/gbert-base",
        "learning_rate": 1e-6,
        "batch_size": 8,
        "num_epochs": 4,
    }
    print("Hyperparams: ", hyperparams)

    tokenizer = BertTokenizer.from_pretrained(hyperparams["model"])
    model = BertForNextSentencePrediction.from_pretrained(
        hyperparams["model"], cache_dir="/raid/6lahann/.cache/huggingface/transformers"
    )
    model = model.to(device)

    # Dataset setup
    auth_token = input("Enter auth token: ").strip()
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"train_balanced": "balanced_train_df.csv"},
        split="train_balanced",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/datasets",
    )
    print(f"No. of examples: {len(dataset)}")
    dataset = dataset.shuffle(seed=6948050)
    dataset = dataset.train_test_split(test_size=0.2, seed=6948050)

    train_ds = dataset["train"]
    train_ds = train_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=32,
        new_fingerprint="train_balanced_2024-02-10 13_55",
    )
    train_ds = train_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )

    val_ds = dataset["test"]
    val_ds = val_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=32,
        new_fingerprint="val_balanced_2024-02-10 13_55",
    )
    val_ds = val_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )

    # training

    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=hyperparams["batch_size"],
        num_train_epochs=hyperparams["num_epochs"],
        learning_rate=hyperparams["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        dataloader_pin_memory=False,
        gradient_accumulation_steps=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
