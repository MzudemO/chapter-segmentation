import numpy as np
from typing import List


# taken from keras to avoid dependency

def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0):
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
