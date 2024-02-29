import datasets
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer
import transformers
import json

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
    output["book_path"] = example["book_path"]
    output["chapter_idx"] = example["chapter_idx"]
    output["paragraph_idx"] = example["paragraph_idx"]
    output["labels"] = torch.tensor(labels, dtype=torch.uint8)
    return output

if __name__ == "__main__":
    results = []
    BATCH_SIZE = 4

    # Model and device setup
    if torch.cuda.device_count() > 0:
        print(
            f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
        )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    auth_token = input("Enter auth token: ").strip()
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained(
        "./test_trainer/checkpoint-22488/",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/transformers",
    )

    model = model.to(device)

    # Dataset setup
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"dataset": "test_per_book_df.csv"},
        split="dataset",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/datasets",
    )

    dataset = dataset.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        new_fingerprint="test_2024-02-20 13_11",
    )
    dataset = dataset.with_format(
        "torch",
        columns=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "labels",
            "book_path",
            "chapter_idx",
            "paragraph_idx",
        ],
        device=device,
    )

    print(len(dataset))

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE
    )

    model.eval()

    for batch in tqdm(dataloader):
        model_batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "token_type_ids", "attention_mask", "labels"] }

        with torch.no_grad():
            outputs = model(**model_batch)

        logits = outputs.logits

        for index, logit in enumerate(logits):
            results.append(
                [
                    batch["book_path"][index],
                    batch["chapter_idx"][index].cpu().item(),
                    batch["paragraph_idx"][index].cpu().item(),
                    batch["labels"][index].cpu().item(),
                    logit[0].cpu().item(),
                    logit[1].cpu().item(),
                ]
            )

    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            0: "book_path",
            1: "chapter_idx",
            2: "paragraph_idx",
            3: "labels",
            4: "logit_0",
            5: "logit_1",
        }
    )
    df.to_pickle("./results/predictions_test.pkl")
