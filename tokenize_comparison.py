import torch
from transformers import BertTokenizer, BatchEncoding
from utils import take_sentences_from_start, take_sentences_from_end
import json
import os
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import pad_sequences
from train import get_tokens

def is_novel(path: str) -> bool:
    with open(f"corpus/{path}", "r", encoding="utf8") as f:
        return "Erzählende Literatur" in json.load(f)["genres"]

def tokenize_sequences(sequence: List, tokenizer: BertTokenizer) -> List:
    p1, p2, is_continuation = sequence
    p1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
    p1 = take_sentences_from_end(p1, 254)
    p2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
    p2 = take_sentences_from_start(p2, 254)
    return [p1, p2, is_continuation]

def tokenize_sequences_new(sequence: List, tokenizer: BertTokenizer) -> Tuple[BatchEncoding, bool]:
    p1, p2, is_continuation = sequence
    p1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
    p1 = take_sentences_from_end(p1, 254)
    p2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
    p2 = take_sentences_from_start(p2, 254)
    batch_encoding = tokenizer.encode_plus(p1, p2, add_special_tokens=True, padding="max_length", truncation=True, max_length=512, return_tensors="np", return_token_type_ids=True, return_attention_mask=True, return_length=True, verbose=True)
    return batch_encoding, is_continuation

def save_split(split: str, paths: List[str]):
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    
    seqs = []
    for path in paths[:1]:
        book = {}
        with open(f"corpus/{path}", "r", encoding="utf8") as f:
            book = json.load(f)


        likely_real_chapters = [c for c in book["chapters"][1:] if c["paragraphs"] != []]
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

    seqs_new = [tokenize_sequences_new(seq, tokenizer) for seq in tqdm(seqs)]
    seqs = [tokenize_sequences(seq, tokenizer) for seq in tqdm(seqs)]
    print(seqs[0])

    df = pd.DataFrame(seqs)
    df = df.rename(columns={0: 'p1_tokens', 1: 'p2_tokens', 2: 'is_continuation'})

    input_tokens = []
    input_segment_ids = []
    labels = []
    for idx, row in tqdm(df.iterrows()):
        indexed_tokens, segments_ids = get_tokens(row['p1_tokens'], row['p2_tokens'], cls, sep)
        input_tokens.append(indexed_tokens)
        input_segment_ids.append(segments_ids)
        labels.append(int(row['is_continuation']))

    print("Created sequences")

    input_ids = pad_sequences(input_tokens, maxlen=512, dtype="long", value=0, truncating="pre", padding="post")
    seg_ids = pad_sequences(input_segment_ids, maxlen=512, dtype="long", value=1, truncating="pre", padding="post")
    attention_masks = [[int(token_id > 0) for token_id in sent] for sent in input_ids]

    print("Input IDs")
    print(input_ids[0])
    print("Input IDs new")
    print(seqs_new[0][0]["input_ids"])
    print("Segment IDs")
    print(seg_ids[0])
    print("Segment IDs new")
    print(seqs_new[0][0]["token_type_ids"])
    print("Attention Masks")
    print(attention_masks[0])
    print("Attention Masks new")
    print(seqs_new[0][0]["attention_mask"])
    print("Length", len(input_ids[0]))
    print("Length New", seqs_new[0][0]["length"])
    print(seqs_new[0][0].keys())

if __name__ == "__main__":

    corpus_files = os.listdir("corpus/")
    corpus_files = list(filter(is_novel, corpus_files))
    print(corpus_files)
    # train, test = train_test_split(corpus_files, train_size=0.8, random_state=6948050)

    save_split("train", corpus_files)
