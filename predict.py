import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from utils import pad_sequences
import os
import json


def predict_single(part_1, part_2, cls, sep, model, tokenizer):



    toks1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(part_1))
    toks2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(part_2))

    toks1 = toks1[-254:]
    toks2 = toks2[:254]

    ids1 = [cls] + toks1 + [sep]
    ids2 = toks2 + [sep]

    indexed_tokens = ids1 + ids2
    segments_ids = [0] * len(ids1) + [1] * len(ids2)

    indexed_tokens = pad_sequences([indexed_tokens], maxlen=512, dtype='long', value=0, truncating="pre", padding="post")
    segments_ids = pad_sequences([segments_ids], maxlen=512, dtype="long", value=1, truncating="pre", padding="post")
    attention_masks = [[int(token_id > 0) for token_id in sent] for sent in indexed_tokens]

    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segments_ids)
    attention_tensor = torch.tensor(attention_masks)

    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    attention_tensor = attention_tensor.to(device)

    model.eval()
    prediction = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)
    prediction = prediction[0]
    softmax = torch.nn.Softmax(dim=1)
    prediction_sm = softmax(prediction)

    print("Same chapter:", (prediction_sm[0, 0] > prediction_sm[0, 1]).item())
    return (prediction_sm[0, 0] > prediction_sm[0, 1]).item()






if __name__ == "__main__":
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('deepset/gbert-base')

    model = BertForNextSentencePrediction.from_pretrained('deepset/gbert-base')
    model = model.to(device)

    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

    for book_name in range(3, 5):
        book = {}
        with open(f"corpus/{book_name}.json", "r", encoding="utf8") as f:
            book = json.load(f)

        seqs = []

        for chapter_index, chapter in enumerate(book["chapters"]):
            for paragraph_index, paragraph in enumerate(chapter["paragraphs"]):
                if chapter_index > 1 and paragraph_index == 0:
                    seqs.append((book["chapters"][chapter_index - 1]["paragraphs"][0], paragraph, False))
                elif paragraph_index > 0:
                    seqs.append((chapter["paragraphs"][paragraph_index - 1], paragraph, True))

        for seq in seqs:
            s1, s2, is_continuation = seq

            pred = predict_single(s1, s2, cls, sep, model, tokenizer)
            print("Predicted chapter break: ", not pred)
            print("True chapter break: ", not is_continuation)
