import json
import stanza

corpus = {}

with open("corpus.json", "r", encoding="utf8") as f:
    corpus = json.load(f)

nlp = stanza.Pipeline("de", processors="tokenize,mwt,pos,lemma")

test_text = corpus[0]["chapters"][1]["paragraphs"][1]

print(test_text)

test_pipeline = nlp(test_text)

print([t.lemma for t in test_pipeline.iter_words()])