LATEX=pdflatex
BIBTEX=bibtex
PAPER=main
PAPER_TEX=$(PAPER).tex
DOC_NAME=$(basename $(PAPER_TEX))
TODO=todo_main
BUILD=build

.PHONY: paper
paper:
	mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=$(DOC_NAME) -output-directory $(BUILD) $(PAPER_TEX) > /dev/null
	$(BIBTEX) $(BUILD)/$(DOC_NAME) > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=$(DOC_NAME) -output-directory $(BUILD) $(PAPER_TEX) > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=$(DOC_NAME) -output-directory $(BUILD) $(PAPER_TEX) > /dev/null
	cp $(BUILD)/$(DOC_NAME).pdf $(DOC_NAME).pdf

.PHONY: paper_main
paper_main: PAPER=main
paper_main: paper

.PHONY: three_agent_eval_note_pdf
three_agent_eval_note_pdf:
	mkdir -p build
	@printf '%s\n' '\documentclass{article}' > build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\usepackage{graphicx}' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\title{Synthetic Evaluation of Temporal Grounding}' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\date{}' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\begin{document}' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\maketitle' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\input{../data_filtering/synthetic_filtering_eval/note/three_agent_eval_note.tex}' >> build/three_agent_eval_note_standalone.tex
	@printf '%s\n' '\end{document}' >> build/three_agent_eval_note_standalone.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=three_agent_eval_note -output-directory build build/three_agent_eval_note_standalone.tex > /dev/null
	$(BIBTEX) build/three_agent_eval_note > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=three_agent_eval_note -output-directory build build/three_agent_eval_note_standalone.tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -jobname=three_agent_eval_note -output-directory build build/three_agent_eval_note_standalone.tex > /dev/null
	cp ../data_filtering/synthetic_filtering_eval/results/plots/three_agent_accuracy_by_question_type.pdf ../data_filtering/synthetic_filtering_eval/note/three_agent_accuracy_by_question_type.pdf
	cp ../data_filtering/synthetic_filtering_eval/results/plots/three_agent_model_pre_post_accuracy.pdf ../data_filtering/synthetic_filtering_eval/note/three_agent_model_pre_post_accuracy.pdf
	cp build/three_agent_eval_note.pdf ../data_filtering/synthetic_filtering_eval/note/three_agent_eval_note.pdf
	rm -f build/three_agent_eval_note_standalone.tex

.PHONY: todo
todo:
	mkdir -p $(BUILD)
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(TODO).tex > /dev/null
	$(LATEX) -interaction=nonstopmode -halt-on-error -synctex=1 -output-directory $(BUILD) $(TODO).tex > /dev/null
	cp $(BUILD)/$(TODO).pdf todo.pdf

.PHONY: clean
clean:
	rm -rf $(BUILD)
