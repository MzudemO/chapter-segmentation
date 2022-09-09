import glob
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import re

import pandas as pd
import numpy as np

books = glob.glob("corpus/*")

all_books = []

for book_path in tqdm(books):
    book = {}
    with open(book_path, "r", encoding="utf8") as f:
        book = json.load(f)

    book_stats = {
        "genres": [re.sub(r"\W+", " ", g) for g in book["genres"]],
        "chapters": [
            {"paragraph_count": len(c["paragraphs"])} for c in book["chapters"]
        ],
    }
    all_books.append(book_stats)

books_df = pd.json_normalize(all_books)

# Find frequent genres
pd.Series(np.concatenate(books_df["genres"])).value_counts()[:20].plot(
    kind="barh", title="20 Most Frequent Genres"
)
plt.savefig("figures/genre_frequencies.png", bbox_inches="tight")

# Count books with genres we don't want
genre_df = books_df.explode(column="genres")
print("Lyrik: ", len(genre_df[genre_df["genres"] == "Lyrik"]))
print("Dramatik: ", len(genre_df[genre_df["genres"] == "Dramatik"]))
print("Anthologie: ", len(genre_df[genre_df["genres"] == "Anthologie"]))
print("Fabeln: ", len(genre_df[genre_df["genres"] == "Fabeln"]))
