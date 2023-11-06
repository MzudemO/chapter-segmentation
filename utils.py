import json
import numpy as np
from typing import List
import pandas as pd
import random
import matplotlib.pyplot as plt
import os


# taken from keras to avoid dependency
def pad_sequences(
    sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0
):
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()

    for x in sequences:
        lengths.append(len(x))
        if len(x):
            sample_shape = np.asarray(x).shape[1:]

    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
    return x


SENTENCE_SEP_TOKENS = [".", "!", "?"]


def take_sentences_from_start(paragraph: List[str], length: int) -> List[str]:
    output = paragraph[:length]
    period_indices = [
        index for index, token in enumerate(output) if token in SENTENCE_SEP_TOKENS
    ]
    if length >= len(paragraph):
        period_indices = period_indices[
            :-1
        ]  # the paragraph ends on a sentence separator
    if period_indices == []:
        return output
    else:
        return output[: period_indices[-1]]


def take_sentences_from_end(paragraph: List[str], length: int) -> List[str]:
    output = paragraph[-length:]
    period_indices = [
        index for index, token in enumerate(output) if token in SENTENCE_SEP_TOKENS
    ]
    period_indices = period_indices[:-1]  # the paragraph ends on a sentence separator
    if period_indices == []:
        return output
    else:
        return output[period_indices[0] + 1 :]


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def is_novel(path: str) -> bool:
    with open(f"corpus/{path}", "r", encoding="utf8") as f:
        return "Erzählende Literatur" in json.load(f)["genres"]


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
