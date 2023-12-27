import re
import urllib
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from utils import filename_from_path
from parse_utils import *
import os
import json
import argparse

LOG_PATH = "parse_log.txt"
TITLEPAGE_INFO = []
TITLEPAGE_WORD_COUNT_CUTOFF = 300
GENRE_LIST = [
    "Romane, Novellen und Erzählungen",
    "Historische Romane und Erzählungen",
    "Anthologien",
    "Romanhafte Biographien",
    "Märchen, Sagen, Legenden",
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


def parse_single_book(work_dict, titlepage_info=False):
    printf = open(LOG_PATH, "a")

    book_path = work_dict["filepath"]
    with open(book_path, "r", encoding="utf-8") as f:
        work_html = BeautifulSoup(f, features="lxml")

    pages = work_html.find(class_="dropdown-content")
    pages = [] if pages == None else pages.findChildren("li")
    pages = [ch.a["href"] for ch in pages]
    # allow for single-page no dropdown works (e.g. Hugo Salus - Der Spiegel)
    pages = [book_path] if len(pages) == 0 else pages
    chapter_idx = 0
    chapters = []

    for page_idx, page in enumerate(pages):
        page_path = os.path.normpath(os.path.join(os.path.dirname(book_path), page))
        with open(page_path, "r", encoding="utf-8") as f:
            page_html = BeautifulSoup(f, features="lxml")

        # skip first page if < 300 words (titlepage, not a real chapter)
        if page_idx == 0:
            raw_paragraphs = page_html.find_all("p")
            raw_paragraphs = [tag_to_text(p) for p in raw_paragraphs]
            raw_paragraphs = [
                p.strip() for p in raw_paragraphs if not p.isspace() and not p == ""
            ]
            word_count = sum([len(p.split(" ")) for p in raw_paragraphs])
            if word_count < TITLEPAGE_WORD_COUNT_CUTOFF:
                continue

        # skip dedication chapters
        if page_path.endswith("dedication.html"):
            continue
        if titlepage_info:
            titlepage_info_dict = {
                "page_idx": page_idx,
                "filepath": page_path,
                "toc": page_html.find(True, class_="toc") is not None,
                "dedication": page_html.find(True, class_="dedication"),
                "titlepage": page_path.endswith("titlepage.html"),
                "dedication": page_path.endswith("dedication.html"),
                "paragraph_stats": [len(p.split(" ")) for p in raw_paragraphs],
            }
            TITLEPAGE_INFO.append(titlepage_info_dict)

        clean_page(page_html)
        chapter_headlines = page_html.find_all(is_headline_before_text)
        # can't safely assume that none-headline pages are new chapters
        # e.g. Christian Reuter - Schelmuffsky, Eufemia von Adlersfeld-Ballestrem - Der Maskenball in der Ca' Torcelli
        # append to previous chapter
        if len(chapter_headlines) == 0:
            paragraphs = page_html.find_all("p")
            paragraphs = [tag_to_text(p) for p in paragraphs]
            paragraphs = [
                p.strip() for p in paragraphs if not p.isspace() and not p == ""
            ]
            if chapter_idx == 0:
                chapter_dict = {
                    "name": None,
                    "idx": chapter_idx,
                    "paragraphs": paragraphs,
                }
                chapters.append(chapter_dict)
                chapter_idx += 1
                continue
            else:
                # concatenate to previous chapter
                # dont increment chapter idx
                chapters[chapter_idx - 1]["paragraphs"] = (
                    chapters[chapter_idx - 1]["paragraphs"] + paragraphs
                )
                continue
        for ch in chapter_headlines:
            chapter_dict = {"name": ch.text, "idx": chapter_idx}
            paragraphs = []
            # iterate through consecutive p-tags
            el = ch.find_next_sibling(True)
            while el != None and el.name == "p":
                paragraphs.append(tag_to_text(el))
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
                        f"SPACER FOUND: {current_el} -> {next_sibling}. Path: {page_path}",
                        file=printf,
                    )

            # filter empty paragraphs
            paragraphs = [
                p.strip() for p in paragraphs if not p.isspace() and not p == ""
            ]
            chapter_dict["paragraphs"] = paragraphs
            chapters.append(chapter_dict)
            chapter_idx += 1

    work_dict["chapters"] = chapters
    printf.close()


def allowed_genre(tag: Tag) -> bool:
    return tag.name == "a" and re.sub(r"\s+", " ", tag.text) in GENRE_LIST


if __name__ == "__main__":
    # cleanup log file
    with open(LOG_PATH, "w") as f:
        f.write("")

    # root_path = "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/info/texte/lesetips.html"
    parser = argparse.ArgumentParser(
        description="Parse dataset from raw GutenbergDE HTML."
    )
    parser.add_argument("base_path", type=str, help="path to the GutenbergDE folder")
    parser.add_argument(
        "--titlepage-info",
        action="store_true",
        help="collect chapter statistics (default: False)",
    )
    parser.add_argument(
        "--test-single", type=str, help="Path to a single book for testing purposes"
    )

    args = parser.parse_args()

    if args.test_single is not None:
        work_dict = {
            "author": "Test Author",
            "title": "Test Title",
            "genre": "Test Genre",
            "webpath": "Test Path",
            "filepath": args.test_single,
        }

        parse_single_book(work_dict)

        with open("test_single_book_parse.json", "w", encoding="utf-8") as f:
            json.dump(work_dict, f, ensure_ascii=False)
    else:
        root_path = os.path.join(args.base_path, "info/texte/lesetips.html")

        with open(root_path, "r", encoding="ISO-8859-1") as f:
            soup = BeautifulSoup(f, features="lxml")

        fiction_genre_list = soup.find("p", string="Belletristik").find_next_sibling(
            "dl"
        )
        genre_links = fiction_genre_list.find_all(allowed_genre)

        work_dicts = []

        for genre_link in genre_links:
            print(len(work_dicts))
            anchor_id = genre_link["href"].split("#")[-1]
            book_list = soup.find("a", id=anchor_id).parent.find_next_sibling("dl")
            current = book_list.find(["dt", "dd"])
            while current != None:
                # skip interspersed book covers
                if current.name == "dt" and current.find("img") != None:
                    current = current.find_next_sibling("dt")
                else:
                    # remove alphabetic marks in Romane, Novellen und Erzählungen
                    alphabetic_marks = current.find_all("b")
                    [am.decompose() for am in alphabetic_marks]
                    author = normalize_author(current.text)
                    book_link = current.find_next_sibling("dd").a
                    title = book_link.text
                    relative_path = book_link["href"]
                    work_dict = {
                        "author": author,
                        "title": title,
                        "genre": re.sub(r"\s+", " ", genre_link.text),
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
            try:
                parse_single_book(work, titlepage_info=args.titlepage_info)
                with open(
                    f'corpus/{filename_from_path(work["filepath"])}',
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(work, f, ensure_ascii=False)
            except UnicodeDecodeError:
                print("DECODE ERROR: ", work["filepath"])
            except FileNotFoundError:
                print("FILE NOT FOUND: ", work["filepath"])

        if args.titlepage_info:
            with open("non_book_chapter_data.json", "w", encoding="utf-8") as f:
                json.dump(TITLEPAGE_INFO, f, ensure_ascii=False)
