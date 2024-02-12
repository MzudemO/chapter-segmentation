import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer, get_scheduler
import transformers
import evaluate
import json

from utils import flat_accuracy


def preprocess(example, tokenizer):
    p1_tokens = list(map(json.loads, example["p1_tokens"]))
    p2_tokens = list(map(json.loads, example["p2_tokens"]))
    sequences = list(zip(p1_tokens, p2_tokens))
    labels = example["is_continuation"]
    batch_encoding = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    output = batch_encoding
    output["labels"] = torch.tensor(labels, dtype=torch.long)
    return output

if __name__ == "__main__":
    BATCH_SIZE = 32
    transformers.logging.set_verbosity_error()  # prevents log spam from false positive warning

    # Model and device setup
    print(
        f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
    )
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    hyperparams = {"model": "deepset/gbert-base", "learning_rate": 2e-05, "batch_size": 4, "num_epochs": 4}
    print("Hyperparams: ", hyperparams)

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
        batch_size=BATCH_SIZE,
        new_fingerprint="train_balanced_2024-02-09 19_34",
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
        batch_size=BATCH_SIZE,
        new_fingerprint="val_balanced_2024-02-09 19_34",
    )
    val_ds = val_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )

    BATCH_SIZE = hyperparams["batch_size"]

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model parameters
    num_epochs = hyperparams["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])

    progress_bar = tqdm(range(num_training_steps))

    training_loss = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            print(outputs[0])
            training_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            progress_bar.update()

        metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        model.eval()

        for batch in val_dataloader:
            print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            print(logits)
            print(logits.shape)
            predictions = torch.argmax(logits, dim=-1)
            print(predictions)
            input("")
            metric.add_batch(predictions=predictions, references=batch["labels"])

        result = metric.compute()

        evaluate.save("./results/", **result, **hyperparams, training_loss=training_loss)

    # Save model
    # model.save_pretrained("./pt_save_pretrained-balanced-e4")
