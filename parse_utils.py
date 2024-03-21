from bs4 import BeautifulSoup, Tag
import re
import os


def filename_from_path(path, extension="json"):
    path_components = path.split(os.sep)
    filename = os.extsep.join([path_components[-2], extension])
    return "_".join(
        [path_components[-3], filename]
    )  # 3rd last is author, 2nd last is work, last sometimes duplicates work


def tag_to_text(tag: Tag) -> str:
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
    return tag.name == "a" and (
        pattern.match(tag.get("id") or "") or pattern.match(tag.get("name") or "")
    )


def replace_with_p(tag: Tag):
    tag.name = "p"
    tag.string = tag_to_text(tag)
    del tag["class"]


# "lastname, firstname" -> "firstname lastname"
def normalize_author(name: str) -> str:
    names = name.split(",")
    names = [n.strip() for n in names]
    names.reverse()
    return " ".join(names)


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

    rulesets = [
        # add letter as single paragraph (Edmond About - Die Spielhölle in Baden-Baden)
        page.find_all("div", class_="letter"),
        # add blockquote as single paragraph (Honoré de Balzac - Glanz und Elend der Kurtisanen)
        page.find_all("blockquote"),
    ]
    for results in rulesets:
        [replace_with_p(r) for r in results]
