LATEX=pdflatex
PAPER=main
BUILD=build

.PHONY: paper
paper:
	mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null

.PHONY: clean
clean:
	rm -rf $(BUILD)
