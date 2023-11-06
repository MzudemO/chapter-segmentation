import requests
from bs4 import BeautifulSoup
import urllib
import re
import glob
from tqdm import tqdm
import json
from utils import filename_from_path

# get downloaded works (differentiate by path)
books = glob.glob("corpus/*.json")

books_without_genres = []

for book_path in tqdm(books):
    with open(book_path, "r", encoding="utf8") as f:
        book = json.load(f)

        if book["genres"] == []:
            books_without_genres.append(book["path"])

# latest archived snapshot with the entire page saved and genres present
root_path = "https://web.archive.org/web/20230319222151/https://www.projekt-gutenberg.org//info/texte/allworka.html"

r = requests.get(root_path)

soup = BeautifulSoup(r.content, features="lxml")

all_works = []
all_works_html = soup.find_all("dd")
print(len(all_works_html))

for work in tqdm(all_works_html):
    link = work.findChild("a")
    if link:
        path = link["href"]
        abspath = urllib.parse.urljoin(root_path, path)
        abspath = re.sub("https://web.archive.org/web/20230319222151/", "", abspath)

        if abspath in books_without_genres:
            genres = work.findChild("i")
            genres = [] if genres == None else genres.text.split(",")
            genres = [s.strip() for s in genres]
            genres = [re.sub(r"\s+", " ", g) for g in genres]

            if genres != []:
                book = {}
                with open(
                    f"corpus/{filename_from_path(abspath)}", "r", encoding="utf8"
                ) as f:
                    book = json.load(f)
                    book["genres"] = genres

                with open(
                    f"corpus/{filename_from_path(abspath)}", "w", encoding="utf8"
                ) as f:
                    json.dump(book, f, ensure_ascii=False)
