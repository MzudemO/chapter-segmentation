import requests
import re
import urllib
import json
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import glob
from utils import filename_from_path
import os


# "lastname, firstname" -> "firstname lastname"
def normalize_author(name: str) -> str:
    names = name.split(",")
    names = [n.strip() for n in names]
    names.reverse()
    return " ".join(names)


root_path = "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/info/texte/lesetips.html"

with open(root_path, "r", encoding="ISO-8859-1") as f:
    soup = BeautifulSoup(f, features="lxml")

genres = ["Über Literatur"]

fiction_genre_list = soup.find("p", string="Belletristik").find_next_sibling("dl")

for genre in genres:
    link = fiction_genre_list.find("a", text=genre)
    if link == None:
        continue
    anchor_id = link["href"].split("#")[-1]
    book_list = soup.find("a", id=anchor_id).parent.find_next_sibling("dl")
    current = book_list.find(["dt", "dd"])
    while current != None:
        # skip interspersed book covers
        if current.name == "dt" and current.find("a") != None:
            current = current.find_next_sibling("dt")
        else:
            author = normalize_author(current.text)
            book_link = current.find_next_sibling("dd").a
            title = book_link.text
            relative_path = book_link["href"]
            work_dict = {
                "author": author,
                "title": title,
                "genre": genre,
                "path": urllib.parse.urljoin(
                    "https://www.projekt-gutenberg.org/info/texte/allworka.html",
                    relative_path,
                ),
            }
            print(work_dict)
            book_path = os.path.normpath(
                os.path.join(os.path.dirname(root_path), relative_path)
            )
            with open(book_path, "r", encoding="ISO-8859-1") as f:
                work_html = BeautifulSoup(f, features="lxml")

            pages = work_html.find(class_="dropdown-content")
            pages = [] if pages == None else pages.findChildren("li")
            pages = [ch.a for ch in pages]

            chapter_idx = 0
            chapters = []

            for page in pages:
                page_path = page["href"]
                page_path = os.path.normpath(
                    os.path.join(os.path.dirname(book_path), page_path)
                )
                with open(page_path, "r", encoding="ISO-8859-1") as f:
                    page_html = BeautifulSoup(f, features="lxml")

                # find h3 or h4 (or others?) where next sibling is <p>
                # should we split anthologies into multiple books?
                # h3 -> book, h4 -> chapter? (if not anthology, h3 -> chapter)

            # continue loop
            current = current.find_next_sibling("dt")


# for work in all_works_html:
#     link = work.findChild("a")
#     if link:
#         path = link["href"]
#         abspath = urllib.parse.urljoin(root_path, path)

#         if abspath not in downloaded_paths:
#             print(abspath)

#             title = re.sub(r"\s+", " ", link.text)
#             genres = work.findChild("i")
#             genres = [] if genres == None else genres.text.split(",")
#             genres = [s.strip() for s in genres]
#             genres = [re.sub(r"\s+", " ", g) for g in genres]

#             all_works.append(
#                 {"path": abspath, "title": title, "genres": genres, "chapters": []}
#             )

# # download all new works
# for work_index, work in enumerate(tqdm(all_works)):
#     try:
#         html = requests.get(work["path"], timeout=10).content
#     except:
#         print(work, work_index)
#         continue
#     work_html = BeautifulSoup(html, features="lxml")

#     chapters = work_html.find(class_="dropdown-content")
#     chapters = [] if chapters == None else chapters.findChildren("li")
#     chapters = [ch.a for ch in chapters]
#     work_chapters = []

#     for idx, chapter in enumerate(chapters):
#         chapter_path = chapter["href"]
#         name = chapter.text
#         index = idx
#         abspath = urllib.parse.urljoin(work["path"], chapter_path)

#         try:
#             html = requests.get(abspath, timeout=10).content
#         except:
#             print(work, work_index)
#             break
#         chapter_html = BeautifulSoup(html, features="lxml")
#         paragraphs = chapter_html.find_all(name="p")
#         paragraphs = [p.text for p in paragraphs]
#         paragraphs = [re.sub(r"\s+", " ", p) for p in paragraphs]
#         paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]

#         work_chapters.append({"name": name, "index": index, "paragraphs": paragraphs})

#     work_dict = work.copy()
#     work_dict["chapters"] = work_chapters

#     with open(
#         f"corpus/{filename_from_path(work_dict['path'])}", "w", encoding="utf8"
#     ) as f:
#         json.dump(work_dict, f, ensure_ascii=False)

#     # rate limit
#     time.sleep(1)
