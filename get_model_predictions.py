import datasets
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer

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
    output["label"] = torch.tensor(labels, dtype=torch.long)
    return output


if __name__ == "__main__":
    results = []
    BATCH_SIZE = 32

    # Model and device setup
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained("./finetuned-balanced-e1")
    model = model.to(device)

    # Dataset setup
    dataset = datasets.load_from_disk("eval-dataset")
    dataset = dataset.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
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

    BATCH_SIZE = 1
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

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

        results.append(
            [
                batch["book"][0],
                batch["chapter"].numpy()[0],
                batch["paragraph"].numpy()[0],
                batch["label"].numpy()[0],
                logits[0][0],
                logits[0][1],
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
    df.to_pickle("results_finetuned-balanced-e1.pkl")
