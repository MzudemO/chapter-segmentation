import glob
import json
from tqdm import tqdm
import pandas as pd
import openpyxl

books = glob.glob("corpus/*.json")

book_stats = []

for book in tqdm(books):
    with open(book, "r", encoding="utf8") as f:
        book = json.load(f)

        book_stats.append(
            {
                "genres": book["genres"],
                "path": book["path"],
                "title": book["title"],
                "chapters": [
                    {"paragraph_count": len(c["paragraphs"])} for c in book["chapters"]
                ],
            }
        )

no_genre = [b for b in book_stats if b["genres"] == []]
print(f"Number of books without a genre: {len(no_genre)}")

df = pd.DataFrame(book_stats)
print(df.head())
df = df[df.genres.apply(len) == 0]
print(df.head())
df = df.drop(columns=["chapters"])
df.to_excel("no_genres.xlsx")
