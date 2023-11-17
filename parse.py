import requests
import re
import urllib
import json
from bs4 import BeautifulSoup, NavigableString, Tag
import time
from tqdm import tqdm
import glob
from utils import filename_from_path
import os
import copy


def p_tag_to_text(tag: Tag) -> str:
    pagerefs = tag.find_all("a", class_="pageref")
    [pr.decompose() for pr in pagerefs]
    return re.sub(r"\s+", " ", tag.text)


def is_headline_before_text(tag: Tag) -> bool:
    return tag.name in ["h2", "h3", "h4"] and tag.find_next_sibling(True).name == "p"


def parse_single_book(work_dict):
    book_path = work_dict["filepath"]
    with open(book_path, "r", encoding="utf-8") as f:
        work_html = BeautifulSoup(f, features="lxml")

    # not present in all?
    pages = work_html.find(class_="dropdown-content")
    pages = [] if pages == None else pages.findChildren("li")
    pages = [ch.a["href"] for ch in pages]
    pages = [book_path] if len(pages) == 0 else pages

    chapter_idx = 0
    chapters = []

    for page in pages:
        page_path = os.path.normpath(os.path.join(os.path.dirname(book_path), page))
        with open(page_path, "r", encoding="utf-8") as f:
            page_html = BeautifulSoup(f, features="lxml")

        chapter_headlines = page_html.find_all(is_headline_before_text)
        for ch in chapter_headlines:
            chapter_dict = {"name": ch.text, "idx": chapter_idx}
            paragraphs = []
            # iterate through consecutive p-tags
            el = ch.find_next_sibling(True)
            while el != None and el.name in ["p", "hr", "div"]:
                class_ = el.get("class")
                class_ = [] if class_ == None else class_
                # TODO: div class "toc" is indicator of titlepage chapter
                # allow for image spacer (e.g. Robert Kraft - Die Vestalinnen, Band 1)
                if el.name == "div" and "figure" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # allow for single hr spacer (e.g. Hugo Salus - Der Spiegel)
                if el.name == "hr":
                    el = el.find_next_sibling(True)
                    continue
                # ignore *** spacer (e.g. Seestern - 1906)
                if el.name == "p" and "stars" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # TODO: check if <br/> pose an issue
                if el.name == "p":
                    paragraphs.append(p_tag_to_text(el))
                    el = el.find_next_sibling(True)

            # element is not P tag
            # for troubleshooting/finding edge cases:
            if el != None:
                current_el = el
                next_sibling = el.find_next_sibling(True)
                if (
                    not is_headline_before_text(current_el)
                    and next_sibling != None
                    and next_sibling.name == "p"
                ):
                    print(
                        f"SPACER FOUND: {current_el} -> {next_sibling}. Path: {page_path}"
                    )

            # filter empty paragraphs
            paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]
            chapter_dict["paragraphs"] = paragraphs
            chapters.append(chapter_dict)
            chapter_idx += 1

    work_dict["chapters"] = chapters


### testing
# work_dict = {
#     "author": "Test Author",
#     "title": "Test Title",
#     "genre": "Test Genre",
#     "path": "Test Path",
# }

# parse_single_book(
#     work_dict,
#     "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/dauthend/novellen/titlepage.html",
# )

# with open("test_single_book_parse.json", "w", encoding="utf-8") as f:
#     json.dump(work_dict, f, ensure_ascii=False)
# input("")


# "lastname, firstname" -> "firstname lastname"
def normalize_author(name: str) -> str:
    names = name.split(",")
    names = [n.strip() for n in names]
    names.reverse()
    return " ".join(names)


root_path = "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/info/texte/lesetips.html"

with open(root_path, "r", encoding="ISO-8859-1") as f:
    soup = BeautifulSoup(f, features="lxml")

genres = [
    "Romane, Novellen und Erzählungen",
    "Historische Romane und Erzählungen",
    "Spannung und Abenteuer",
    "Krimis, Thriller, Spionage",
    "Historische Kriminalromane und -fälle",
    "Science Fiction",
    "Phantastische Literatur",
    "Horror",
    "Humor, Satire",
    "Kinderbücher bis 11 Jahre",
    "Kinderbücher ab 12 Jahren",
]

fiction_genre_list = soup.find("p", string="Belletristik").find_next_sibling("dl")

for genre in genres:
    work_dicts = []
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
                "webpath": urllib.parse.urljoin(
                    "https://www.projekt-gutenberg.org/info/texte/allworka.html",
                    relative_path,
                ),
                "filepath": os.path.normpath(
                    os.path.join(os.path.dirname(root_path), relative_path)
                ),
            }
            work_dicts.append(work_dict)

            # continue loop
            current = current.find_next_sibling("dt")

    print(len(work_dicts))
    for work in tqdm(work_dicts):
        parse_single_book(work_dict)
        with open(
            f'corpus/{filename_from_path(work_dict["filepath"])}', "w", encoding="utf-8"
        ) as f:
            json.dump(work_dict, f, ensure_ascii=False)
