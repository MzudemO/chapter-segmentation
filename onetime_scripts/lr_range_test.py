import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer, get_scheduler
import transformers
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from torch import nn
import json
import matplotlib.pyplot as plt


def preprocess(example, tokenizer):
    p1_tokens = list(map(json.loads, example["p1_tokens"]))
    p2_tokens = list(map(json.loads, example["p2_tokens"]))
    sequences = list(zip(p1_tokens, p2_tokens))
    labels = example["is_continuation"]
    batch_encoding = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    output = batch_encoding
    output["labels"] = torch.tensor(labels, dtype=torch.long)
    return output


class LossWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, _labels):
        # we already pass labels into BERT model and get loss back
        return outputs.loss


class BertModelWrapper(nn.Module):
    def __init__(self, bert_model, device=None):
        super(BertModelWrapper, self).__init__()
        self.bert_model = bert_model
        self.device = device

    def _move_to_device(self, encoded_inputs):
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

    def forward(self, batch, labels=None):
        outputs = self.bert_model(**batch)
        return outputs


class TrainDataLoaderWrapper(TrainDataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def inputs_labels_from_batch(self, batch_data):
        labels = batch_data["labels"]
        return (batch_data, labels)

class ValDataLoaderWrapper(ValDataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def inputs_labels_from_batch(self, batch_data):
        labels = batch_data["labels"]
        return (batch_data, labels)


if __name__ == "__main__":
    BATCH_SIZE = 32
    transformers.logging.set_verbosity_error()  # prevents log spam from false positive warning

    # Model and device setup
    if torch.cuda.device_count() > 0:
        print(
            f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
        )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained(
        "deepset/gbert-base",
        cache_dir="/raid/6lahann/.cache/huggingface/transformers"
    )
    model = model.to(device)

    # Dataset setup
    auth_token = input("Enter auth token: ").strip()
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"train_balanced": "balanced_train_df.csv"},
        split="train_balanced",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/datasets",
    )
    print(f"No. of examples: {len(dataset)}")
    dataset = dataset.shuffle(seed=6948050)
    dataset = dataset.train_test_split(test_size=0.2, seed=6948050)

    train_ds = dataset["train"]
    train_ds = train_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        new_fingerprint="lr_range_test",
    )
    train_ds = train_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )

    val_ds = dataset["test"]
    val_ds = val_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        new_fingerprint="lr_range_test_val",
    )
    val_ds = val_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        device=device,
    )

    BATCH_SIZE = 8

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    train_data_loader_wrapper = TrainDataLoaderWrapper(train_dataloader)
    
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    val_data_loader_wrapper = ValDataLoaderWrapper(val_dataloader)

    model_wrapper = BertModelWrapper(model, device)
    loss_wrapper = LossWrapper()

    num_epochs = 3
    # num_iter = num_epochs * len(train_dataloader)
    num_iter = 50
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_finder = LRFinder(model_wrapper, optimizer, loss_wrapper, device=device)
    lr_finder.range_test(train_data_loader_wrapper, val_loader=val_data_loader_wrapper, end_lr=1, num_iter=num_iter, diverge_th=10, step_mode="linear")

    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax, suggest_lr=True, log_lr=False)
    plt.savefig("figures/lr_range_linear.svg")
    with open("lr_range_test_linear.json", "w") as f:
        json.dump(lr_finder.history, f)

    lr_finder.reset()
