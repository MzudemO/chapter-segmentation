import datasets
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer
import transformers
from utils import preprocess

if __name__ == "__main__":
    results = []
    BATCH_SIZE = 8

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
        "MzudemO/chapter-segmentation-model",
        revision="752630c190621d1cf28350bb83dff2f0d7749344",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/transformers",
    )

    model = model.to(device)

    # Dataset setup
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"dataset": "test_for_eval.csv"},
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
            "label",
            "book_path",
            "chapter_idx",
            "paragraph_idx",
        ],
        device=device,
    )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=0
    )

    model.eval()

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits

        for index, logit in enumerate(logits):
            results.append(
                [
                    batch["book"][index],
                    batch["chapter"][index].cpu().item(),
                    batch["paragraph"][index].cpu().item(),
                    batch["label"][index].cpu().item(),
                    logit[0],
                    logit[1],
                ]
            )

    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            0: "book_path",
            1: "chapter_idx",
            2: "paragraph_idx",
            3: "label",
            4: "logit_0",
            5: "logit_1",
        }
    )
    df.to_pickle("./results/results_finetuned-full.pkl")
