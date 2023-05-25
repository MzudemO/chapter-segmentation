import torch
from transformers import BertTokenizer
from utils import take_sentences_from_start, take_sentences_from_end, is_novel
import json
import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def tokenize_sequences(sequence: List, tokenizer: BertTokenizer) -> List:
    book_i, chapter_i, paragraph_i, p1, p2, is_continuation = sequence
    p1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
    p1 = take_sentences_from_end(p1, 254)
    p2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
    p2 = take_sentences_from_start(p2, 254)
    return [book_i, chapter_i, paragraph_i, p1, p2, is_continuation]


def save_evaluation_split(split: str, paths: List[str]):
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    seqs = []
    for path in tqdm(paths):
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
                            path,
                            chapter_index,
                            paragraph_index,
                            likely_real_chapters[chapter_index - 1]["paragraphs"][-1],
                            paragraph,
                            False,
                        ]
                    )
                elif paragraph_index > 0:
                    seqs.append(
                        [
                            path,
                            chapter_index,
                            paragraph_index,
                            chapter["paragraphs"][paragraph_index - 1],
                            paragraph,
                            True,
                        ]
                    )

    seqs = [tokenize_sequences(seq, tokenizer) for seq in tqdm(seqs)]

    df = pd.DataFrame(seqs)
    df = df.rename(
        columns={
            0: "book",
            1: "chapter",
            2: "paragraph",
            3: "p1_tokens",
            4: "p2_tokens",
            5: "is_continuation",
        }
    )

    print(df.head(5))

    df.to_pickle(f"{split}_df.pkl")


if __name__ == "__main__":

    corpus_files = os.listdir("corpus/")
    corpus_files = list(filter(is_novel, corpus_files))
    _, test = train_test_split(corpus_files, train_size=0.8, random_state=6948050)
    save_evaluation_split("test_for_eval", test)
