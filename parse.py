import re
import urllib
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from utils import filename_from_path
import os
import json


def p_tag_to_text(tag: Tag) -> str:
    pagerefs = tag.find_all("a", class_="pageref")
    [pr.decompose() for pr in pagerefs]
    return re.sub(r"\s+", " ", tag.text)


def is_headline_before_text(tag: Tag) -> bool:
    next_sibling = tag.find_next_sibling(True)
    if tag.name in ["h2", "h3", "h4"] and next_sibling is None:
        return False
    else:
        return tag.name in ["h2", "h3", "h4"] and next_sibling.name == "p"


def is_empty_or_nonword_p(tag: Tag) -> bool:
    if tag.name == "p":
        return tag.text.isspace() or re.search(r"\w", tag.text) is None
    else:
        return False


def is_pagelink(tag: Tag) -> bool:
    pattern = re.compile("^page.+$")
    if tag.get("id") != None and tag.get("name") != None:
        return tag.name == "a" and (
            pattern.match(tag.get("id")) or pattern.match(tag.get("name"))
        )
    else:
        return False


def clean_page(page: BeautifulSoup) -> None:
    rulesets = [
        page.find_all("a", class_="pageref"),
        page.find_all(is_pagelink),
        # non-chapter breaking headlines (Edmondo De Amicis - Unsere Freunde)
        page.find_all("h5"),
        # non-chapter breaking headlines (Max Bartel - Aufstieg der Begabten)
        page.find_all("h6"),
        # non-chapter breaking headlines (Rudolf Herzog - Wieland der Schmied)
        page.find_all("h1"),
        # poetry/lyrics (Honoré de Balzac - Lebensbilder - Band 1)
        page.find_all("p", class_="vers"),
        # breaks (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Alkiphron - Hetärenbriefe)
        page.find_all("br"),
        # single hr spacer (Hugo Salus - Der Spiegel)
        page.find_all("hr"),
        # *** spacer (Seestern - 1906)
        page.find_all("p", class_="stars"),
        # short poetry and other tables (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Roland Betsch - Der Wilde Freiger)
        page.find_all("table"),
        # image spacer (Robert Kraft - Die Vestalinnen, Band 1)
        page.find_all("div", class_="figure"),
        # poetry/lyrics (Ulrich Hegner - Die Molkenkur)
        page.find_all("div", class_="poem"),
        # motto (Scholem Alejchem - Aus dem nahen Osten)
        page.find_all("div", class_="motto"),
        # poster (Arkadij Awertschenko - Kurzgeschichten)
        page.find_all("div", class_="plakat"),
        # poetry/lyrics (Otto Julius Bierbaum - Sinaide)
        page.find_all("div", class_="vers"),
        # box formatting (Georg Weerth - Leben und Taten des berühmten Ritters Schnapphahnski)
        page.find_all("div", class_="box"),
        # irregular formatted text (Georg Weerth - Leben und Taten des berühmten Ritters Schnapphahnski)
        page.find_all("pre"),
        # in-text footnotes (Petronius - Begebenheiten des Enkolp)
        page.find_all("span", class_="footnote"),
        # images (Hanns Heiz Ewers - Grotesken)
        page.find_all("img"),
        # lists (Karl Adolph - Haus Nummer 37)
        page.find_all("ol"),
        # lists (Charles Dickens - Klein-Dorrit. Zweites Buch)
        page.find_all("ul"),
        # aside images (Emanuel Friedli - Bärndütsch als Spiegel bernischen Volkstums / Vierter Band)
        page.find_all("aside"),
        # address (Adelheid von Auer - Fußstapfen im Sande. Erster Band)
        page.find_all("address"),
        # * spacer, ...
        page.find_all(is_empty_or_nonword_p),
    ]
    for results in rulesets:
        [r.decompose() for r in results]


def parse_single_book(work_dict):
    printf = open("parse_log.txt", "a")

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

    for page in pages:
        page_path = os.path.normpath(os.path.join(os.path.dirname(book_path), page))
        with open(page_path, "r", encoding="utf-8") as f:
            page_html = BeautifulSoup(f, features="lxml")

        clean_page(page_html)
        if page_path.endswith("titlepage.html"):
            print("Titlepage: ", len(page_html.find_all("p")), page_path, file=printf)
            continue
        chapter_headlines = page_html.find_all(is_headline_before_text)
        # can't safely assume that none-headline pages are new chapters
        # e.g. Christian Reuter - Schelmuffsky, Eufemia von Adlersfeld-Ballestrem - Der Maskenball in der Ca' Torcelli
        # marked clearly in data with "name": None
        # up to user whether to consider same or separate
        if len(chapter_headlines) == 0:
            chapter_dict = {"name": None, "idx": chapter_idx}
            paragraphs = page_html.find_all("p")
            paragraphs = [p_tag_to_text(p) for p in paragraphs]
            paragraphs = [
                p.strip() for p in paragraphs if not p.isspace() and not p == ""
            ]
            chapter_dict["paragraphs"] = paragraphs
            chapters.append(chapter_dict)
            chapter_idx += 1
            print(f"No-headline chapter: {page_path}", file=printf)
            continue
        for ch in chapter_headlines:
            chapter_dict = {"name": ch.text, "idx": chapter_idx}
            paragraphs = []
            # iterate through consecutive p-tags
            el = ch.find_next_sibling(True)
            while el != None and el.name in ["p", "div", "blockquote"]:
                class_ = el.get("class")
                class_ = [] if class_ == None else class_
                # TODO: div class "toc" is indicator of titlepage chapter
                # add letter as single paragraph (Edmond About - Die Spielhölle in Baden-Baden)
                if el.name == "div" and "letter" in class_:
                    paragraphs.append(p_tag_to_text(el))
                    el = el.find_next_sibling(True)
                # add blockquote as single paragraph (Honoré de Balzac - Glanz und Elend der Kurtisanen)
                elif el.name == "blockquote":
                    paragraphs.append(p_tag_to_text(el))
                    el = el.find_next_sibling(True)
                elif el.name == "p":
                    paragraphs.append(p_tag_to_text(el))
                    el = el.find_next_sibling(True)
                else:
                    break

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


# testing
# work_dict = {
#     "author": "Test Author",
#     "title": "Test Title",
#     "genre": "Test Genre",
#     "webpath": "Test Path",
#     "filepath": "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/reuterc/schelmuf/titlepage.html",
# }

# parse_single_book(work_dict)

# with open("test_single_book_parse.json", "w", encoding="utf-8") as f:
#     json.dump(work_dict, f, ensure_ascii=False)
# print("done")
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
]

fiction_genre_list = soup.find("p", string="Belletristik").find_next_sibling("dl")

work_dicts = []

for genre in genres:
    link = fiction_genre_list.find("a", text=genre)
    if link == None:
        continue
    anchor_id = link["href"].split("#")[-1]
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
    try:
        parse_single_book(work)
        with open(
            f'corpus/{filename_from_path(work["filepath"])}', "w", encoding="utf-8"
        ) as f:
            json.dump(work, f, ensure_ascii=False)
    except UnicodeDecodeError:
        print("DECODE ERROR: ", work["filepath"])
    except FileNotFoundError:
        print("FILE NOT FOUND: ", work["filepath"])
