import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForNextSentencePrediction, BertTokenizer
import transformers
import evaluate

from utils import flat_accuracy


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
    return output


if __name__ == "__main__":
    BATCH_SIZE = 32
    transformers.logging.set_verbosity_error() # prevents log spam from false positive warning

    # Model and device setup
    print(
        f"Devices available: {torch.cuda.device_count()}. Device 0: {torch.cuda.get_device_name(0)}."
    )
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    model = BertForNextSentencePrediction.from_pretrained(
        "./pt_save_pretrained-balanced-e1", 
        # cache_dir="/raid/6lahann/.cache/huggingface/transformers"
    )
    model = model.to(device)

    # Dataset setup
    auth_token = input("Enter auth token: ").strip()
    dataset = datasets.load_dataset(
        "MzudemO/ger-chapter-segmentation",
        data_files={"test": "test.csv"},
        split="test",
        use_auth_token=auth_token,
        cache_dir="/raid/6lahann/.cache/huggingface/datasets",
    )
    print(f"No. of examples: {len(dataset)}")

    test_ds = dataset
    test_ds = test_ds.map(
        lambda example: preprocess(example, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        new_fingerprint="test_2023-01-13 13_18"
    )
    test_ds = test_ds.with_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        device=device,
    )

    BATCH_SIZE = 32

    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # Testing
    
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in test_dataloader:
        batch = (
            batch["input_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch["label"],
        )
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_token_type_ids, b_attention_masks, b_labels = batch
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_masks,
            )

        logits = outputs[0]

        metric.add_batch(
            predictions=torch.argmax(logits, dim=1), references=b_labels
        )
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

    result = metric.compute()
    hyperparams = {"model": "deepset/gbert-base-pretrained-balanced-e1"}
    evaluate.save("./results/", **result, **hyperparams)
