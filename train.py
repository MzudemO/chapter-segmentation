import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForNextSentencePrediction
from utils import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
import numpy as np

def get_tokens(p1, p2, cls, sep):
    p1_tokens = [cls] + p1 + [sep]
    p2_tokens = p2 + [sep]

    tokens = p1_tokens + p2_tokens
    segment_ids = [0] * len(p1_tokens) + [1] * len(p2_tokens)

    return tokens, segment_ids

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained("deepset/gbert-base")
    model = model.to(device)

    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

    df = pd.read_pickle("test_df.pkl")

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

    print("Padded sequences")

    train_input_ids, validation_input_ids, train_seg_ids, validation_seg_ids, train_attention_masks, validation_attention_masks, train_labels, validation_labels = train_test_split(input_ids, seg_ids, attention_masks, labels, random_state=6948050, test_size=0.1)

    print("Generated train/val split")

    train_input_ids = torch.tensor(train_input_ids)
    validation_input_ids = torch.tensor(validation_input_ids)
    train_seg_ids = torch.tensor(train_seg_ids)
    validation_seg_ids = torch.tensor(validation_seg_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    validation_attention_masks = torch.tensor(validation_attention_masks)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    BATCH_SIZE = 32

    train_data = TensorDataset(train_input_ids, train_seg_ids, train_attention_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    
    validation_data = TensorDataset(validation_input_ids, validation_seg_ids, validation_attention_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    print("Created dataloaders")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
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

            outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks, next_sentence_label=b_labels) 

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
                outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))