import spacy

from spacy.tokens import Doc, DocBin

import json
import glob
from tqdm import tqdm

nlp = spacy.load("de_core_news_sm", disable=["attribute_ruler", "ner"])
nlp.disable_pipe("parser")
nlp.enable_pipe("senter")

Doc.set_extension("chapter_index", default=None)
Doc.set_extension("paragraph_index", default=None)

books = glob.glob("corpus/*")

for book_path in tqdm(books):
    docbin = DocBin(store_user_data=True)

    book_name = book_path.split("/")[1].split(".")[0]
    book = {}
    with open(f"corpus/{book_name}.json", "r", encoding="utf8") as f:
        book = json.load(f)

    for c_index, chapter in enumerate(book["chapters"]):
        for p_index, paragraph in enumerate(chapter["paragraphs"]):
            doc = nlp(paragraph)
            doc._.chapter_index = c_index
            doc._.paragraph_index = p_index

            docbin.add(doc)

    docbin.to_disk(f"corpus_spacy/{book_name}.spacy")
