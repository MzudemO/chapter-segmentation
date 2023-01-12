import pandas as pd

train_df = pd.read_pickle("train_df.pkl")
chapter_breaks = train_df[train_df["is_continuation"] == False]
nr_chapter_breaks = len(chapter_breaks)
print(nr_chapter_breaks)


continuations = train_df[train_df["is_continuation"] == True]
continuations = continuations.sample(n=nr_chapter_breaks, random_state=6948050)

new_df = pd.concat([chapter_breaks, continuations])
new_df = new_df.sample(frac=1, random_state=6948050).reset_index(drop=True)

print(len(new_df))
print(new_df.head())

new_df.to_csv("train_balanced.csv")
