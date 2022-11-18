import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BatchEncoding
from typing import Tuple
from utils import take_sentences_from_start, take_sentences_from_end
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from transformers import BertTokenizer, BertForNextSentencePrediction
from utils import pad_sequences
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def tokenize_sequence(
    sequence: pd.DataFrame, tokenizer: BertTokenizer
) -> Tuple[BatchEncoding, bool]:
    p1 = sequence["p1_tokens"]
    p2 = sequence["p2_tokens"]
    label = sequence["is_continuation"]
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
    return batch_encoding, label


class BertSequenceDataset(Dataset):
    def __init__(self, sequences: pd.DataFrame, tokenizer: BertTokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        batch_encoding, label = tokenize_sequence(sequence, self.tokenizer)
        input_ids = batch_encoding["input_ids"][0]
        segment_ids = batch_encoding["token_type_ids"][0]
        attention_mask = batch_encoding["attention_mask"][0]
        return input_ids, segment_ids, attention_mask, label


if __name__ == "__main__":
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained("deepset/gbert-base")
    model = model.to(device)

    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

    train_df = pd.read_pickle("test_df.pkl")
    train = train_df.sample(frac=0.1, random_state=6948050)
    val = train_df.drop(train.index)

    train_data = BertSequenceDataset(train, tokenizer)
    validation_data = BertSequenceDataset(val, tokenizer)

    print("Created datasets")

    BATCH_SIZE = 32

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE
    )

    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE
    )

    print("Created dataloaders")

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    print("Making model GPU compatible")
    model = model.to(device)

    epochs = 1
    loss_values = []
    total_loss = 0

    for epoch in range(epochs):
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch

            optimizer.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=b_seg_ids,
                attention_mask=b_attention_masks,
                next_sentence_label=b_labels,
            )

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_seg_ids,
                    attention_mask=b_attention_masks,
                )

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
