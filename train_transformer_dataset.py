import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer
import transformers
import evaluate

from utils import flat_accuracy


def preprocess(example, tokenizer):
    sequences = zip(example["p1_tokens"], example["p2_tokens"])
    labels = example["is_continuation"]
    batch_encoding = tokenizer.batch_encode_plus(
        sequences,
        # add_special_tokens=True, seems to be a bug with the argument handling
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    output = batch_encoding
    output["label"] = torch.tensor(labels, dtype=torch.long)
    return output


if __name__ == "__main__":
    BATCH_SIZE = 32
    transformers.logging.set_verbosity_error() # prevents log spam from false positive warning

    # Model and device setup
    print(
        f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
    )
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained(
        "deepset/gbert-base", cache_dir="/raid/6lahann/.cache/huggingface/transformers"
    )
    model = model.to(device)

    # Dataset setup
    auth_token = input("Enter auth token: ").strip()
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"train_balanced": "train_balanced.csv"},
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
        new_fingerprint="train_balanced_2023-01-12 16_22"
    )
    train_ds = train_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        device=device,
    )

    val_ds = dataset["test"]
    val_ds = val_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        new_fingerprint="2023-01-12 16_22"
    )
    val_ds = val_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        device=device,
    )

    BATCH_SIZE = 8

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    epochs = 4
    loss_values = []
    total_loss = 0

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # Training loop
    for epoch in range(epochs):
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = (
                batch["input_ids"],
                batch["token_type_ids"],
                batch["attention_mask"],
                batch["label"],
            )
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_attention_masks, b_labels = batch

            optimizer.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_masks,
                labels=b_labels,
            )

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in val_dataloader:
            batch = (
                batch["input_ids"],
                batch["token_type_ids"],
                batch["attention_mask"],
                batch["label"],
            )
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_masks,
                )

            logits = outputs[0]

            metric.add_batch(
                predictions=torch.argmax(logits, dim=1), references=b_labels
            )
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

    result = metric.compute()
    hyperparams = {"model": "deepset/gbert-base"}
    evaluate.save("./results/", **result, **hyperparams)

    # Save model
    model.save_pretrained("./pt_save_pretrained-balanced-e4")
