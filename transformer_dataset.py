from datasets import Dataset
import pandas as pd

train_df = pd.read_pickle("train_df.pkl")

train_dataset = Dataset.from_pandas(train_df)

print(train_dataset.features)

train_dataset.save_to_disk("./ger-chapter-segmentation")
