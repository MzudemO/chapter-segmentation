import requests
import re
import urllib
import json
from bs4 import BeautifulSoup

# wget -c --random-wait -r -p --regex-type pcre --accept-regex 'gutenberg.org/[0-9\-a-z]+/([0-9\-a-z]+)/(.*).html$' -U mozilla

root_path = "https://www.projekt-gutenberg.org/info/texte/allworka.html"

r = requests.get(root_path)

soup = BeautifulSoup(r.content, features="lxml")

all_works = []
all_works_html = soup.find_all("dd")[:1]

for work in all_works_html:
    link = work.findChild("a")
    path = link["href"]
    title = re.sub(r"\W+", " ", link.text)
    genres = work.findChild("i").text.split(",")
    genres = [s.strip() for s in genres]

    print(f"{title} ({genres}) - {path}")

    abspath = urllib.parse.urljoin(root_path, path)
    print(f"Absolute Path: {abspath}")

    all_works.append({"path": abspath, "title": title, "genres": genres, "chapters": []})

for work in all_works:
    work_html = BeautifulSoup(requests.get(work["path"]).content, features="lxml")
    
    chapters = work_html.find(class_="dropdown-content").findChildren("li")
    chapters = [ch.a for ch in chapters]

    for idx, chapter in enumerate(chapters):
        chapter_path = chapter["href"]
        name = chapter.text
        index = idx
        abspath = urllib.parse.urljoin(work["path"], chapter_path)

        chapter_html = BeautifulSoup(requests.get(abspath).content, features="lxml")
        paragraphs = chapter_html.find_all(name="p")
        paragraphs = [p.text for p in paragraphs]
        paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]

        work["chapters"].append({"name": name, "index": index, "paragraphs": paragraphs})

with open("corpus.json", "w", encoding="utf8") as f:
    json.dump(all_works, f, ensure_ascii=False)