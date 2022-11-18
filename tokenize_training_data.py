import torch
from transformers import BertTokenizer, BatchEncoding
from utils import take_sentences_from_start, take_sentences_from_end
import json
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


def is_novel(path: str) -> bool:
    with open(f"corpus/{path}", "r", encoding="utf8") as f:
        return "Erzählende Literatur" in json.load(f)["genres"]


def tokenize_sequences(
    sequence: List, tokenizer: BertTokenizer
) -> Tuple[BatchEncoding, bool]:
    p1, p2, is_continuation = sequence
    p1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
    p1 = take_sentences_from_end(p1, 254)
    p2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
    p2 = take_sentences_from_start(p2, 254)
    batch_encoding = tokenizer.encode_plus(
        p1,
        p2,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    return batch_encoding, is_continuation


def save_split(split: str, paths: List[str]):
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    seqs = []
    for path in paths:
        book = {}
        with open(f"corpus/{path}", "r", encoding="utf8") as f:
            book = json.load(f)

        likely_real_chapters = [
            c for c in book["chapters"][1:] if c["paragraphs"] != []
        ]
        if len(likely_real_chapters) < 2:
            print("Not enough chapters in", path)
            continue

        for chapter_index, chapter in enumerate(likely_real_chapters):
            for paragraph_index, paragraph in enumerate(chapter["paragraphs"]):
                if chapter_index > 0 and paragraph_index == 0:
                    seqs.append(
                        [
                            likely_real_chapters[chapter_index - 1]["paragraphs"][-1],
                            paragraph,
                            False,
                        ]
                    )
                elif paragraph_index > 0:
                    seqs.append(
                        [chapter["paragraphs"][paragraph_index - 1], paragraph, True]
                    )

    seqs = [tokenize_sequences(seq, tokenizer) for seq in tqdm(seqs)]

    torch.save(seqs, f"{split}_data.pt")


if __name__ == "__main__":
    corpus_files = os.listdir("corpus/")
    corpus_files = list(filter(is_novel, corpus_files))
    train, test = train_test_split(corpus_files, train_size=0.8, random_state=6948050)

    # train_splits = np.array_split(np.array(train), 10)

    # for i, split in enumerate(train_splits):
    #     save_split(f"train_{i}", split)

    test_splits = np.array_split(np.array(test), 4)

    for i, split in enumerate(test_splits):
        save_split(f"test_{i}", split)
