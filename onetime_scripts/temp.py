import pandas as pd
from tqdm import tqdm

line_count = 0

with pd.read_csv("train_df.csv", engine="c", chunksize=10**5) as reader:
    for chunk in tqdm(reader):
        line_count += len(chunk)


print(line_count)
