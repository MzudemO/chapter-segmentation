# Remove

* pageref (Roland Betsch - Der Wilde Freiger)
* table (Roland Betsch - Der Wilde Freiger, Willibald Alexis - Ruhe ist die erste Bürgerpflicht)
* no headline & first chapter could be title page? (Abraham a Sancta Clara - Verschiedenes, Karl Adolph - Von früher und heute, Alexandra, königliche Prinzessin von Bayern - Maiglöckchen)
    - check if no p tag with text at all -> safe bet
    - 1st pass: remove all empty p tags? alejchem/nahosten/chap002.html
* headline detection: 
    - table between headline & chapter (Willibald Alexis - Walladmor)
    - `<hr class="short"/>` (Henri Allais - Die Wachsbüste und andere Erzählungen)
* `<br clear="all"/>` (Willibald Alexis - Ruhe ist die erste Bürgerpflicht, Alkiphron - Hetärenbriefe)
* `<h5>` (Edmondo De Amicis - Unsere Freunde, Theodor Herzl - Philosophische Erzählungen, H. Clauren - Die Versuchung)
* `<ol>` (Karl Adolph - Haus Nummer 37, Schalom Alechem - Die erste jüdische Republik, Honoré de Balzac - Louis Lambert)
* `<p class="vers>` (Honoré de Balzac - Lebensbilder - Band 1)
* `<h1>` (Rudolf Herzog - Wieland der Schmied)
* `<a id="page###" name="page###">` various
* `<div class="poem">` (Ulrich Hegner - Die Molkenkur)

# Keep

* letter as 1 paragraph 
    - /about/badenbad/chap003.html 
    - /achleitn/moor/chap005.html 
    - adolph/toechter/chap010.html 
    - salome/dashaus/chap015.html 
    - anet/kleinsta/chap003.html 
    - anet/lydiaser/chap004.html 
    - balzac/kurtisan/chap001.html
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
* fucked up encoding & weird structure:
    - Hans Christian Andersen (Nur ein Geiger, Der Glücks-Peter, Der Improvisator, Oz, Sein oder Nichtsein)
    - Restif de la Bretonne (Anti-Justine)
* missing link:
    - Honoré de Balzac - Die Messe der Gottlosen