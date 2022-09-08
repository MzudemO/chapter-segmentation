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

pd.Series(np.concatenate(books_df["genres"])).value_counts()[:20].plot(
    kind="barh", title="20 Most Frequent Genres"
)
plt.savefig("figures/genre_frequencies.png", bbox_inches="tight")
