# Remove

* pageref (Roland Betsch - Der Wilde Freiger)
* table (Roland Betsch - Der Wilde Freiger, Willibald Alexis - Ruhe ist die erste Bürgerpflicht)
* no headline & first chapter could be title page? (Abraham a Sancta Clara - Verschiedenes, Karl Adolph - Von früher und heute, Alexandra, königliche Prinzessin von Bayern - Maiglöckchen)
    - `div class="toc">`?
    - `div class="dedication">`?
    - "titlepage" in url: 
        - some counterexamples (Christoph Martin Wieland - Nachlaß des Diogenes von Sinope, Berthold Auerbach - Barfüßele, Ludwig Bechstein - Der Dunkelgraf, Daniel Defoe - Moll Flanders, Marie von Olfers - Frost in Blüthen)
        - some are title pages with higher paragraph count (Hans Bethge - Der gelbe Kater, Edward Lytton Bulwer - Die Caxctons. Band I, Alphonse Daudet - Tartarin in den Alpen [has TOC])
        - mix of paragraph count & paragraph length?
        - duplicated content on titlepage (F.M. Dostojewski - Erniedrigte und Beleidigte, Das Gut Stepantschikowo und seine Bewohner, Aufzeichnungen aus einem toten Hause)
* headline detection: 
    - table between headline & chapter (Willibald Alexis - Walladmor)
    - `<hr class="short"/>` (Henri Allais - Die Wachsbüste und andere Erzählungen)
* `<br clear="all"/>` (Willibald Alexis - Ruhe ist die erste Bürgerpflicht)
* `<h5>` (Edmondo De Amicis - Unsere Freunde, Theodor Herzl - Philosophische Erzählungen, H. Clauren - Die Versuchung)
* `<ol>` (Karl Adolph - Haus Nummer 37, Schalom Alechem - Die erste jüdische Republik, Honoré de Balzac - Louis Lambert)
* `<ul>` (Charles Dickens - Klein-Dorrit. Zweites Buch)
* `<p class="vers>` (Honoré de Balzac - Lebensbilder - Band 1)
* `<h1>` (Rudolf Herzog - Wieland der Schmied, Gustav Aimard - Mexikanische Nächte Zweiter Theil, Wilkie Collins - Blinde Liebe. Zweiter Band, Louis Couperus - Herakles, Hans Fallada - Der eiserne Gustav)
* `<a id="page###" name="page###">` various (e.g. Edward Bulwer-Lytton - Godolphin oder der Schwur, Gustav Meyrink - Das grüne Gesicht)
* `<div class="poem">` (Ulrich Hegner - Die Molkenkur)
* `<div class="motto">` (Scholem Alejchem - Aus dem nahen Osten)
* `<div class="box">` (Georg Weerth - Leben und Taten des berühmten Ritters Schnapphahnski)
* `<address>` (Adelheid von Auer - Fußstapfen im Sande. Erster Band, Edward Bulwer-Lytton - Meine Novelle. Erster Band)
* `<div class="plakat">` (Arkadij Awertschenko - Kurzgeschichten)
* `<h6>` (Max Bartel - Aufstieg der Begabten, Otto Blumenthal - Eine Frauenbeichte und Anderes, Jakob Boßhardt - Ein Rufer in der Wüste)
* `<div class="vers">` (Otto Julius Bierbaum - Sinaide)
* `<img>` (Hanns Heiz Ewers - Grotesken, Hans Fallada - Märchen vom Stadtschreiber, der aufs Land flog)
* `<aside>` (Emanuel Friedli - Bärndütsch als Spiegel bernischen Volkstums / Vierter Band)
* `<pre>` (Georg Weerth - Leben und Taten des berühmten Ritters Schnapphahnski, Jules Verne - Die Kinder des Kapitän Grant. Erster Band)
* `<span class="footnote">` (Petronius - Begebenheiten des Enkolp)
* empty or completely non-word `<p>` (Henri Allais - Die Wachsbüste und andere Erzählungen, Scholem Alejchem - Aus dem nahen Osten)


# Keep

* letter as 1 paragraph 
    - /about/badenbad/chap003.html 
    - /achleitn/moor/chap005.html 
    - adolph/toechter/chap010.html 
    - salome/dashaus/chap015.html 
    - anet/kleinsta/chap003.html 
    - anet/lydiaser/chap004.html 
    - balzac/kurtisan/chap001.html
    - zolling/million/chap017.html
* blockquote as 1 paragraph (hundreds of occurences, want to reduce jumps in text) 
    - about/badenbad/chap008.html
    - balzac/kurtisan/chap003.html
    - balzac/lebensb1/chap02.html (counterexample)

# Other

* non-chapter breaking spacers: stars, horizontal rule (Hugo Salus - Der Spiegel, Seestern - 1906)
* multiple chapters on one page (Hugo Salus - Der Spiegel)
* multiple books (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Italienische Novellen I - III)
    - treat as single book w/ multiple chapters
    - no hierarchical distinction
    - reasoning: segmentation based on narrative structure, decided by author
    - maybe filter short chapters?
    - easier or more difficult?
* assume each page is at least one chapter, even if no headline?
    - Eufemia von Adlersfeld-Ballestrem - Der Maskenball in der Ca' Torcelli (unreliable source - narrative breaks)
    - Eufemia von Adlersfeld-Ballestrem - Das Rosazimmer (pages = *** spacers, but very long 20+ pages)
    - Willibald Alexis - Die Hosen des Herrn von Bredow (chapters clearly marked > 1 page)
    - Wilhelm Heinse - Ardinghello und die glückseligen Inseln (No chapters, parts clearly marked > 1 page)
    - Georg Hermann - Doktor Herzfeld. Erster Teil. Die Nacht (No chapters)
    - Georg Hermann - Grenadier Wordelmann (unreliable source)
    - Elisabeth von Heyking - Ille Mihi (unreliable source)
    - Paul Heyse - L'Arrabbiata (no chapters, novella)
    - Paul Heyse - Der Kinder Sünde der Väter Fluch (no chapters, novella)
    - Paul Oscar Höcker - Die Verbotene Frucht (no clearly marked chapters)
    - Eduard Graf von Keyserling - Die dritte Stiege (chapters clearly marked > 1 page)
    - Eduard Graf von Keyserling - Am Südhang (no clearly marked chapters)
    - Johannes Kirschweng - Der Schäferkarren (chapters = pages)
    - Klabund - Bracke (no clearly marked chapters)
    - Jeremias Gotthelf - Wie Uli der Knecht glücklich wird (chapters clearly marked > 1 page)
    - Paul Grabein - Nomaden (no source - narrative breaks)
    - Ravi Ravendro - Tanzende Flamme (no clearly marked chapters)
    - Christian Reuter - Schelmuffsky (chapters clearly marked > 1 page)
    - Theodor Storm - Curator (no clearly marked chapters)
    - Jean Paul - Das Kampaner Tal (chapters clearly marked > 1 page)
* fucked up encoding & weird structure:
    - Hans Christian Andersen (Nur ein Geiger, Der Glücks-Peter, Der Improvisator, Oz, Sein oder Nichtsein)
    - Restif de la Bretonne - Anti-Justine
    - Alkiphron - Hetärenbriefe
* missing link:
    - Honoré de Balzac - Die Messe der Gottlosen