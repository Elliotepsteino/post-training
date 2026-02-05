LATEX=pdflatex
TEX=filtering

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	$(LATEX) $(TEX).tex > /dev/null
	$(LATEX) $(TEX).tex > /dev/null

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).pdf $(TEX).toc

.PHONY: all clean
