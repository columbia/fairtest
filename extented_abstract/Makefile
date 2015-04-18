PAPER=paper
LATEX=pdflatex
BIBTEX=bibtex

LATEXARGS=
BIBTEXARGS=
PYDFLATEX = $(shell which pydflatex 2>/dev/null | grep -v "aliased" | head -1)

SRCS=$(wildcard *.tex)
REFS=$(wildcard *.bib)
FIGS=$(wildcard figures/*)

ifneq ($(PYDFLATEX),)
  LATEX = $(PYDFLATEX)
  LATEXARGS = -l
  BIBTEX = $(PYDFLATEX)
  BIBTEXARGS = --bibtex
endif

.PHONY: results $(PAPER).pdf clean

all: $(PAPER).pdf

$(PAPER).pdf: $(SRCS) $(REFS) $(FIGS)
	$(LATEX) $(LATEXARGS) $(PAPER)
	$(BIBTEX) $(BIBTEXARGS) $(PAPER)
	$(LATEX) $(LATEXARGS) $(PAPER)
	$(LATEX) $(PAPER)


submission: $(PAPER).pdf
	cp $(PAPER).pdf submission/

clean:
	rm -f *.dvi *.aux *.log *.blg *.bbl \
          $(PAPER).ps $(PAPER).pdf \
          $(PAPER)-single.pdf $(PAPER)-single.ps
