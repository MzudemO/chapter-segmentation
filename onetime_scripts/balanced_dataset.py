import pandas as pd
from tqdm import tqdm

dfs = []

with pd.read_csv("train_df.csv", engine="c", chunksize=10**6) as reader:
    for chunk in tqdm(reader):
        chapter_breaks = chunk[chunk["is_continuation"] == False]
        nr_chapter_breaks = len(chapter_breaks)
        print(nr_chapter_breaks)
        dfs.append(chapter_breaks)

        continuations = chunk[chunk["is_continuation"] == True]
        print(len(continuations))
        continuations = continuations.sample(n=nr_chapter_breaks, random_state=6948050)
        dfs.append(continuations)

balanced_df = pd.concat(dfs)
balanced_df = balanced_df.sample(frac=1, random_state=6948050).reset_index(drop=True)

print(balanced_df.head())
print(len(balanced_df))
print(len(balanced_df[balanced_df["is_continuation"] == False]))

balanced_df.to_csv("balanced_train_df.csv")
