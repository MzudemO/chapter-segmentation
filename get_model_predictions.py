import datasets
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer
import transformers

from utils import flat_accuracy


def preprocess(example, tokenizer):
    sequences = zip(example["p1_tokens"], example["p2_tokens"])
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
    output["book"] = example["book"]
    output["chapter"] = example["chapter"]
    output["paragraph"] = example["paragraph"]
    labels = example["is_continuation"]
    output["label"] = example["label"]
    return output


if __name__ == "__main__":
    results = []
    BATCH_SIZE = 8
    transformers.logging.set_verbosity_error() # prevents log spam from false positive warning
    torch.set_num_threads(1) # try to prevent 100% cpu usage

    # Model and device setup
    print(
        f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
    )
    auth_token = input("Enter auth token: ").strip()
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained(
        "MzudemO/chapter-segmentation-model", 
        revision="752630c190621d1cf28350bb83dff2f0d7749344", 
        use_auth_token=auth_token, 
        cache_dir="/raid/6lahann/.cache/huggingface/transformers"
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
        new_fingerprint="test_2023-05-26 01_40"
    )
    dataset = dataset.with_format(
        "torch",
        columns=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "label",
            "book",
            "chapter",
            "paragraph",
        ],
        device=device,
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=0)

    model.eval()

    for batch in tqdm(dataloader):
        d_batch = (
            batch["input_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch["label"],
        )
        d_batch = tuple(t.to(device) for t in d_batch)
        b_input_ids, b_token_type_ids, b_attention_masks, b_labels = d_batch
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_masks,
            )

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        
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
            0: "book",
            1: "chapter",
            2: "paragraph",
            3: "label",
            4: "logit_0",
            5: "logit_1",
        }
    )
    df.to_pickle("./results/results_finetuned-full.pkl")
