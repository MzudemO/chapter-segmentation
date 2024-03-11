from transformers import BertTokenizerFast
import json
import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def tokenize_sequence(sequence: List, tokenizer: BertTokenizerFast) -> List:
    p1, p2, is_continuation = sequence
    p1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
    p2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
    return [p1[-254:], p2[:254], is_continuation]


def save_split(
    split: str, paths: List[str], tokenizer: BertTokenizerFast, is_test: bool = False
):
    df = pd.DataFrame(
        {
            "p1_tokens": [],
            "p2_tokens": [],
            "is_continuation": [],
        }
    )
    if is_test:
        df["book_path"] = []
        df["chapter_idx"] = []
        df["paragraph_idx"] = []
    df.to_csv(f"{split}_df.csv", index=False, header=True, mode="w")
    for path in tqdm(paths):
        with open(f"corpus/{path}", "r", encoding="utf8") as f:
            book = json.load(f)

        for chapter_index, chapter in enumerate(book["chapters"]):
            paragraphs = chapter["paragraphs"]
            for paragraph_index, paragraph in enumerate(paragraphs):
                if chapter_index > 0 and paragraph_index == 0:
                    previous_paragraph = " ".join(
                        book["chapters"][chapter_index - 1]["paragraphs"]
                    )
                    previous_paragraph = " ".join(previous_paragraph.split(" ")[-300:])
                    paragraph = " ".join(paragraphs[paragraph_index : len(paragraphs)])
                    paragraph = " ".join(paragraph.split(" ")[:300])
                    sequence = tokenize_sequence(
                        [
                            previous_paragraph,
                            paragraph,
                            False,
                        ],
                        tokenizer,
                    )
                    df = pd.DataFrame(
                        {
                            "p1_tokens": [sequence[0]],
                            "p2_tokens": [sequence[1]],
                            "is_continuation": [sequence[2]],
                        }
                    )
                    if is_test:
                        df["book_path"] = path
                        df["chapter_idx"] = chapter_index
                        df["paragraph_idx"] = paragraph_index
                    df.to_csv(f"{split}_df.csv", index=False, header=False, mode="a")

                elif paragraph_index > 0:
                    previous_paragraph = " ".join(paragraphs[0:paragraph_index])
                    previous_paragraph = " ".join(previous_paragraph.split(" ")[-300:])
                    paragraph = " ".join(paragraphs[paragraph_index : len(paragraphs)])
                    paragraph = " ".join(paragraph.split(" ")[:300])
                    sequence = tokenize_sequence(
                        [previous_paragraph, paragraph, True], tokenizer
                    )
                    df = pd.DataFrame(
                        {
                            "p1_tokens": [sequence[0]],
                            "p2_tokens": [sequence[1]],
                            "is_continuation": [sequence[2]],
                        }
                    )
                    if is_test:
                        df["book_path"] = path
                        df["chapter_idx"] = chapter_index
                        df["paragraph_idx"] = paragraph_index
                    df.to_csv(f"{split}_df.csv", index=False, header=False, mode="a")
                    if (
                        len(previous_paragraph.split(" ")) > 254
                        and len(sequence[0]) < 254
                    ):
                        print("P1 WARNING")
                    if len(paragraph.split(" ")) > 254 and len(sequence[1]) < 254:
                        print("P2 WARNING")


if __name__ == "__main__":
    corpus_files = os.listdir("corpus/")
    corpus_files.sort()
    train, test = train_test_split(
        corpus_files, train_size=0.8, shuffle=True, random_state=6948050
    )
    tokenizer = BertTokenizerFast.from_pretrained("deepset/gbert-base")
    print(len(train))
    print(len(test))
    save_split("train", train, tokenizer)
    save_split("test_per_book", test, tokenizer, is_test=True)
