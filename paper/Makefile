LATEX=pdflatex
BIBTEX=bibtex
PAPER=main
TODO=todo_main
BUILD=build

.PHONY: paper
paper:
	mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null
	$(BIBTEX) $(BUILD)/$(PAPER) > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(PAPER).tex > /dev/null
	cp $(BUILD)/$(PAPER).pdf $(PAPER).pdf

.PHONY: todo
todo:
	mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(TODO).tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(TODO).tex > /dev/null
	cp $(BUILD)/$(TODO).pdf todo.pdf

.PHONY: clean
clean:
	rm -rf $(BUILD)
