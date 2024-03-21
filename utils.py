import json
import numpy as np
from typing import List
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import torch


def plot_book(dataframe: pd.DataFrame, book_id=None, fig_size=(20, 5)):
    book_id = book_id
    if book_id == None:
        book_id = random.choice(dataframe["book"])
    book_df = dataframe[dataframe["book"] == book_id]
    book_df.index = [i - min(book_df.index) for i in book_df.index]
    candidates = book_df.sort_values("logit_0", ascending=False)[
        : len(book_df[book_df["label"] == 0])
    ]

    plt.figure(figsize=fig_size)
    plt.title(f"BERT-predicted chapter breaks - {book_id}")
    plt.xlabel("Paragraph")
    plt.ylabel("BERT confidence score")
    plt.scatter(book_df.index, book_df["logit_0"], s=5)
    plt.scatter(candidates.index, candidates["logit_0"], c="red", marker="s")
    xcoords = book_df[book_df["label"] == 0].index
    for x in xcoords:
        plt.axvline(x, c="green")
    plt.show()


def filename_from_path(path, extension="json"):
    path_components = path.split(os.sep)
    filename = os.extsep.join([path_components[-2], extension])
    return "_".join(
        [path_components[-3], filename]
    )  # 3rd last is author, 2nd last is work, last sometimes duplicates work


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
    output["labels"] = torch.tensor(labels, dtype=torch.uint8)
    return output
