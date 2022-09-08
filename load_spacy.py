import spacy
from spacy.tokens import Token, Doc, DocBin
import glob


def filter_token(token: Token) -> bool:
    if token.is_stop or token.is_punct or token.is_space:
        return False
    else:
        return True


nlp = spacy.load("de_core_news_sm", disable=["attribute_ruler", "ner"])
Doc.set_extension("chapter_index", default=None)
Doc.set_extension("paragraph_index", default=None)

books = glob.glob("corpus_spacy/*")[:5]

for book_path in books:
    print(book_path)
    docs = DocBin().from_disk(book_path).get_docs(nlp.vocab)

    for paragraph in docs:
        print(paragraph._.chapter_index)
        print(paragraph._.paragraph_index)
        for sent in paragraph.sents:
            filtered = filter(filter_token, sent)
            # for t in filtered:
            #     if t.lemma_ == "--":
            #         print(t.text_with_ws)

            print([t.lemma_ for t in filtered])
