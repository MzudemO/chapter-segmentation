import datasets
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def preprocess(example, tokenizer):
    sequences = zip(example["p1_tokens"], example["p2_tokens"])
    labels = example["is_continuation"]
    batch_encoding = tokenizer.batch_encode_plus(
        sequences,
        # add_special_tokens=True, seems to be a bug with the argument handling
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    output = batch_encoding
    output["label"] = torch.tensor(labels, dtype=torch.long)
    # print(output)
    # input("lol")
    return output


if __name__ == "__main__":
    BATCH_SIZE = 32

    # Model and device setup
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained("deepset/gbert-base")
    model = model.to(device)

    # Dataset setup
    dataset = datasets.load_from_disk("ger-chapter-segmentation")
    dataset = dataset.shuffle(seed=6948050)
    dataset = dataset.train_test_split(train_size=0.001, seed=6948050)

    train_ds = dataset["train"]
    train_ds = train_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
    )
    train_ds = train_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        device=device,
    )

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Model parameters
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

    epochs = 1
    loss_values = []
    total_loss = 0

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # Training loop
    for epoch in range(epochs):
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = (
                batch["input_ids"],
                batch["token_type_ids"],
                batch["attention_mask"],
                batch["label"],
            )
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_attention_masks, b_labels = batch

            # print(b_input_ids)
            # break

            optimizer.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_masks,
                labels=b_labels,
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

        # for batch in validation_dataloader:
        #     batch = tuple(t.to(device) for t in batch)
        #     b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch
        #     with torch.no_grad():
        #         outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks)

        #     logits = outputs[0]

        #     logits = logits.detach().cpu().numpy()
        #     label_ids = b_labels.to('cpu').numpy()

        #     tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #     eval_accuracy += tmp_eval_accuracy
        #     nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))