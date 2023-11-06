import os
import glob
from utils import filename_from_path
from tqdm import tqdm
import json

books = glob.glob("corpus/*.json")


for book_path in tqdm(books):
    book = {}
    with open(book_path, "r", encoding="utf8") as f:
        book = json.load(f)

    os.rename(book_path, f"corpus/{filename_from_path(book['path'])}")
