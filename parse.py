import requests
import re
import urllib
import json
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# wget -c --random-wait -r -p --regex-type pcre --accept-regex 'gutenberg.org/[0-9\-a-z]+/([0-9\-a-z]+)/(.*).html$' -U mozilla

root_path = "https://www.projekt-gutenberg.org/info/texte/allworka.html"

r = requests.get(root_path)

soup = BeautifulSoup(r.content, features="lxml")

all_works = []
all_works_html = soup.find_all("dd")
print(len(all_works_html))

for work in all_works_html:
    link = work.findChild("a")
    if link:
        path = link["href"]
        # correct broken url
        path = (
            path if path.startswith("..") else "../../antholog/scheusal/scheusal.html"
        )
        title = re.sub(r"\s+", " ", link.text)
        genres = work.findChild("i")
        genres = [] if genres == None else genres.text.split(",")
        genres = [s.strip() for s in genres]
        genres = [re.sub(r"\s+", " ", g) for g in genres]

        abspath = urllib.parse.urljoin(root_path, path)

        all_works.append(
            {"path": abspath, "title": title, "genres": genres, "chapters": []}
        )

for work_index, work in enumerate(tqdm(all_works)):
    work_html = BeautifulSoup(requests.get(work["path"]).content, features="lxml")

    chapters = work_html.find(class_="dropdown-content")
    chapters = [] if chapters == None else chapters.findChildren("li")
    chapters = [ch.a for ch in chapters]
    work_chapters = []

    for idx, chapter in enumerate(chapters):
        chapter_path = chapter["href"]
        name = chapter.text
        index = idx
        abspath = urllib.parse.urljoin(work["path"], chapter_path)

        chapter_html = BeautifulSoup(requests.get(abspath).content, features="lxml")
        paragraphs = chapter_html.find_all(name="p")
        paragraphs = [p.text for p in paragraphs]
        paragraphs = [re.sub(r"\s+", " ", p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]

        work_chapters.append({"name": name, "index": index, "paragraphs": paragraphs})

    work_dict = work.copy()
    work_dict["chapters"] = work_chapters

    with open(f"corpus/{work_index}.json", "w", encoding="utf8") as f:
        json.dump(work_dict, f, ensure_ascii=False)

    # rate limit
    time.sleep(0.5)
