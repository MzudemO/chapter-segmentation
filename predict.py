import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from utils import pad_sequences
import json
from typing import List

SENTENCE_SEP_TOKENS = [".", "!", "?"]
GENRE_DENYLIST = ["Musik", "Lyrik", "Cartoons"]


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


def predict_single(
    part_1: str,
    part_2: str,
    cls: int,
    sep: int,
    model: BertForNextSentencePrediction,
    tokenizer: BertTokenizer,
) -> bool:
    toks1 = tokenizer.tokenize(part_1)
    toks2 = tokenizer.tokenize(part_2)

    toks1 = take_sentences_from_end(toks1, 254)
    toks2 = take_sentences_from_start(toks2, 254)
    # toks1 = toks1[-254:]
    # toks2 = toks2[:254]

    print("S1:", " ".join(toks1))
    print("S2:", " ".join(toks2))

    toks1 = tokenizer.convert_tokens_to_ids(toks1)
    toks2 = tokenizer.convert_tokens_to_ids(toks2)

    ids1 = [cls] + toks1 + [sep]
    ids2 = toks2 + [sep]

    indexed_tokens = ids1 + ids2
    segments_ids = [0] * len(ids1) + [1] * len(ids2)

    indexed_tokens = pad_sequences(
        [indexed_tokens],
        maxlen=512,
        dtype="long",
        value=0,
        truncating="pre",
        padding="post",
    )
    segments_ids = pad_sequences(
        [segments_ids],
        maxlen=512,
        dtype="long",
        value=1,
        truncating="pre",
        padding="post",
    )
    attention_masks = [
        [int(token_id > 0) for token_id in sent] for sent in indexed_tokens
    ]

    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segments_ids)
    attention_tensor = torch.tensor(attention_masks)

    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    attention_tensor = attention_tensor.to(device)

    prediction = model(
        tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor
    )
    prediction = prediction[0]
    softmax = torch.nn.Softmax(dim=1)
    prediction_sm = softmax(prediction)

    # print("Same chapter:", (prediction_sm[0, 0] > prediction_sm[0, 1]).item())
    return (
        (prediction_sm[0, 0] > prediction_sm[0, 1]).item(),
        (abs(prediction_sm[0, 0] - prediction_sm[0, 1])).item(),
    )


if __name__ == "__main__":
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained("deepset/gbert-base")
    model = model.to(device)
    model.eval()

    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

    predictions = []

    for book_name in [0, 1]:
        book = {}
        with open(f"corpus/{book_name}.json", "r", encoding="utf8") as f:
            book = json.load(f)

        if any(item in book["genres"] for item in GENRE_DENYLIST):
            continue

        print(book["genres"])
        print(book["title"])
        seqs = []

        for chapter_index, chapter in enumerate(book["chapters"]):
            for paragraph_index, paragraph in enumerate(chapter["paragraphs"]):
                if chapter_index > 1 and paragraph_index == 0:
                    seqs.append(
                        (
                            book["chapters"][chapter_index - 1]["paragraphs"][-1],
                            paragraph,
                            False,
                        )
                    )
                elif paragraph_index > 0:
                    seqs.append(
                        (chapter["paragraphs"][paragraph_index - 1], paragraph, True)
                    )

        for seq in seqs:
            s1, s2, is_continuation = seq

            pred_continuation, confidence = predict_single(
                s1, s2, cls, sep, model, tokenizer
            )
            # print("Predicted chapter break:", not pred_continuation)
            # print("True chapter break:", not is_continuation)
            # print("Confidence:", confidence)
            # print("---")

            predictions.append((is_continuation, pred_continuation))

    correct_continuation = 0
    false_continuation = 0
    correct_break = 0
    false_break = 0

    for gt, pred in predictions:
        if pred == gt == True:
            correct_continuation += 1
        elif pred == True and gt == False:
            false_continuation += 1
        elif pred == gt == False:
            correct_break += 1
        elif pred == False and gt == True:
            false_break += 1

    print(
        f"""Correctly guessed same chapter: {correct_continuation}.
            Falsely guessed same chapter: {false_continuation}.
            Correctly guessed chapter break: {correct_break}.
            Falsely guessed chapter break: {false_break}.
            Total correct guesses: {correct_continuation + correct_break}.
            Total percentage: {(correct_continuation + correct_break) / len(predictions) * 100}%
            Percentage of chapter breaks guessed correctly: {correct_break / (correct_break + false_continuation) * 100}%"""
    )
