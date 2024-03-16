from typing import List
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.signal import argrelmin, find_peaks, argrelmax
import math


def gather_split(split: str):
    nltk.download("punkt")

    df_list = []
    corpus_files = os.listdir("./corpus/")
    corpus_files.sort()
    train, test = train_test_split(
        corpus_files, train_size=0.8, shuffle=True, random_state=6948050
    )
    splits = {"train": train, "test": test}
    for file in splits[split]:
        with open(f"./corpus/{file}", "r", encoding="utf8") as f:
            d = json.load(f)
            df = pd.DataFrame(d)
            df = df.drop(columns=["author", "webpath", "title"])
            df["filepath"] = pd.Series([file for _ in range(len(df))])
            df_list.append(df)

    df = pd.concat(df_list)
    print(df.head())

    grouped_df = df.groupby(by=["filepath"])
    books_df_list = []
    for name, group in tqdm(grouped_df):
        word_sum = sentence_sum = paragraph_sum = 0
        for chapter in group["chapters"]:
            word_count = sum(
                [
                    len(word_tokenize(p, language="german"))
                    for p in chapter["paragraphs"]
                ]
            )
            word_sum += word_count
            sentence_count = sum(
                [
                    len(sent_tokenize(p, language="german"))
                    for p in chapter["paragraphs"]
                ]
            )
            sentence_sum += sentence_count
            paragraph_count = len(chapter["paragraphs"])
            paragraph_sum += paragraph_count

        books_df_list.append(
            pd.DataFrame(
                {
                    "paragraph_count": [paragraph_sum],
                    "word_count": [word_sum],
                    "sentence_count": [sentence_sum],
                    "book_path": [name],
                    "chapter_count": len(group["chapters"]),
                }
            )
        )

    stats_df = pd.concat(books_df_list)
    stats_df.to_pickle(f"per_chapter_{split}_split_stats.pkl")


def gather_text_metrics(split: str) -> pd.DataFrame:
    split_file = f"per_chapter_{split}_split_stats.pkl"
    if not os.path.isfile(split_file):
        gather_split(split)

    return pd.read_pickle(split_file)


def train_model(df: pd.DataFrame, columns: List[str]) -> linear_model.LinearRegression:
    x = df[columns].to_numpy()
    y = np.array(df["chapter_count"]).reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=6948050
    )
    print(len(x_train))
    print(len(x_test))
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)
    return model


def predict(
    model: linear_model.LinearRegression, df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    x = df[columns].to_numpy()
    y_pred = model.predict(x)
    pred_df = df[["book_path", "chapter_count"]].copy()
    pred_df["predicted_chapter_count"] = y_pred.flatten()
    return pred_df


def filter_novels(df: pd.DataFrame) -> pd.DataFrame:
    novels = []
    for book in df["book_path"]:
        with open(f"./corpus/{book}", "r") as f:
            genre = json.load(f)["genre"]
        if genre == "Romane, Novellen und Erzählungen":
            novels.append(book)

    return df[df["book_path"].isin(novels)]


def confidence_df() -> pd.DataFrame:
    pred_df = pd.read_pickle("./results/predictions_finetuned_2e-5_bs32_e3_best.pkl")

    confidence_stats = []
    for name, group in pred_df.groupby(["book_path"]):
        chapter_count = group["chapter_idx"].max()  # num chapter breaks
        confidences = np.array(group["logit_0"])
        maxima = argrelmax(confidences)[0]
        candidates = group[group["logit_0"] > group["logit_1"]]
        normalized_logit_0 = (group["logit_0"] - group["logit_0"].min()) / (
            group["logit_0"].max() - group["logit_0"].min()
        )
        log_confidence = [-math.log(1 - l + 1e-10) for l in normalized_logit_0]
        log_threshold_candidates = [c > 0.99 for c in log_confidence]
        confidence_stats.append(
            {
                "book_path": name,
                "chapter_count": chapter_count,
                "maxima_count": len(maxima),
                "candidate_count": len(candidates),
                "log_candidate_count": len(log_threshold_candidates),
            }
        )

    return pd.DataFrame(confidence_stats)


if __name__ == "__main__":
    train_stats_df = gather_text_metrics("train")
    test_stats_df = gather_text_metrics("test")

    metrics_model = train_model(
        train_stats_df, ["word_count", "sentence_count", "paragraph_count"]
    )
    metrics_pred_df = predict(
        metrics_model,
        test_stats_df,
        ["word_count", "sentence_count", "paragraph_count"],
    )

    train_novel_stats_df = filter_novels(train_stats_df)

    novels_metrics_model = train_model(
        train_novel_stats_df, ["word_count", "sentence_count", "paragraph_count"]
    )
    novels_metrics_pred_df = predict(
        novels_metrics_model,
        test_stats_df,
        ["word_count", "sentence_count", "paragraph_count"],
    )

    confidences_df = confidence_df()

    train_novels, _ = train_test_split(
        confidences_df["book_path"], test_size=0.2, shuffle=True, random_state=6948050
    )
    confidences_train_df = confidences_df[
        confidences_df["book_path"].isin(train_novels)
    ]

    confidences_model = train_model(
        confidences_train_df, ["maxima_count", "candidate_count", "log_candidate_count"]
    )
    confidences_pred_df = predict(
        confidences_model,
        confidences_df,
        ["maxima_count", "candidate_count", "log_candidate_count"],
    )

    confidences_novel_train_df = filter_novels(confidences_train_df)

    confidences_novel_model = train_model(
        confidences_novel_train_df,
        ["maxima_count", "candidate_count", "log_candidate_count"],
    )
    confidences_novel_pred_df = predict(
        confidences_novel_model,
        confidences_df,
        ["maxima_count", "candidate_count", "log_candidate_count"],
    )

    pd.DataFrame(
        {
            "book_path": test_stats_df["book_path"].to_numpy(),
            "chapter_count": test_stats_df["chapter_count"].to_numpy(),
            "metrics_prediction": metrics_pred_df["predicted_chapter_count"].to_numpy(),
            "novel_metrics_prediction": novels_metrics_pred_df[
                "predicted_chapter_count"
            ].to_numpy(),
            "confidences_prediction": confidences_pred_df[
                "predicted_chapter_count"
            ].to_numpy(),
            "novel_confidences_prediction": confidences_novel_pred_df[
                "predicted_chapter_count"
            ].to_numpy(),
        }
    ).to_pickle("test_predicted_chapter_counts.pkl")
