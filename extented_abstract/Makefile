FILE = paper

PRINTER = lp
UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
	OPEN_COMMAND = evince
endif
ifeq ($(UNAME),Darwin)
	OPEN_COMMAND = open
endif

all:	$(FILE).tex bib
	rm -f $(FILE).ps
	latex $(FILE)
	dvips $(FILE).dvi -o $(FILE).ps
#	$(OPEN_COMMAND) $(FILE).ps

clean:
	rm -f $(FILE).ps $(FILE).pdf $(FILE).log $(FILE).dvi $(FILE).bbl $(FILE).blg $(FILE).aux
	rm -f *~
	rm -f *.bak
	rm -f *.backup
	rm -f *.log

pdf:
	gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=$(FILE).pdf $(FILE).ps -c quit

bib:	$(FILE).tex
	latex $(FILE)
	bibtex $(FILE)
	latex $(FILE)
	latex $(FILE)
	dvips $(FILE).dvi -o $(FILE).ps
	rm $(FILE).dvi $(FILE).log $(FILE).blg

s:	$(FILE).tex
	latex $(FILE)
	dvips $(FILE).dvi -o $(FILE).ps

f:	s
	ps2pdf $(FILE).ps

p:	
	lpr -P$(PRINTER) $(FILE).ps; 
	lpq -P$(PRINTER)

dist:
	tar cfj ../$(FILE)-`date +%F`.tar.bz2 .
