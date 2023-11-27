import re
import urllib
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from utils import filename_from_path
import os


def p_tag_to_text(tag: Tag) -> str:
    pagerefs = tag.find_all("a", class_="pageref")
    [pr.decompose() for pr in pagerefs]
    return re.sub(r"\s+", " ", tag.text)


def is_headline_before_text(tag: Tag) -> bool:
    next_sibling = tag.find_next_sibling(True)
    if tag.name in ["h2", "h3", "h4"] and next_sibling is None:
        print("NEXT SIBLING IS NONE", tag)
        return False
    else:
        return tag.name in ["h2", "h3", "h4"] and next_sibling.name == "p"


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

        chapter_headlines = page_html.find_all(is_headline_before_text)
        # random sample tests show we can assume non-titled chapters to be new chapters
        if len(chapter_headlines) == 0:
            chapter_dict = {"name": None, "idx": chapter_idx}
            paragraphs = page_html.find_all("p")
            paragraphs = [p_tag_to_text(p) for p in paragraphs]
            paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]
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
            while el != None and el.name in ["p", "hr", "div", "table"]:
                class_ = el.get("class")
                class_ = [] if class_ == None else class_
                # TODO: div class "toc" is indicator of titlepage chapter
                # allow for breaks (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Alkiphron - Hetärenbriefe)
                if el.name == "br":
                    el = el.find_next_sibling(True)
                    continue
                # allow for image spacer (Robert Kraft - Die Vestalinnen, Band 1)
                elif el.name == "div" and "figure" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # allow for single hr spacer (Hugo Salus - Der Spiegel)
                elif el.name == "hr":
                    el = el.find_next_sibling(True)
                    continue
                # ignore *** spacer (Seestern - 1906)
                elif el.name == "p" and "stars" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # skip short poetry and other tables (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Roland Betsch - Der Wilde Freiger)
                elif el.name == "table":
                    el = el.find_next_sibling(True)
                    continue
                # skip poetry/lyrics (Honoré de Balzac - Lebensbilder - Band 1)
                elif el.name == "p" and "vers" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # skip poetry/lyrics (Ulrich Hegner - Die Molkenkur)
                elif el.name == "div" and "poem" in class_:
                    el = el.find_next_sibling(True)
                    continue
                # skip lists (Karl Adolph - Haus Nummer 37)
                elif el.name == "ol":
                    el = el.find_next_sibling(True)
                    continue
                # add letter as single paragraph (Edmond About - Die Spielhölle in Baden-Baden)
                elif el.name == "div" and "letter" in class_:
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
            paragraphs = [p for p in paragraphs if not p.isspace() and not p == ""]
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
#     "filepath": "/mnt/c/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Master's Thesis/gutenberg-edition16/adlersfe/maskenba/maskenba.html",
# }

# parse_single_book(work_dict)

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
    except UnicodeDecodeError:
        print("DECODE ERROR: ", work["filepath"])
    # with open(
    #     f'corpus/{filename_from_path(work_dict["filepath"])}', "w", encoding="utf-8"
    # ) as f:
    #     json.dump(work_dict, f, ensure_ascii=False)
