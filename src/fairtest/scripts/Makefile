FIGS = race_discrimination_on_location_dependency\
       race_discrimination_on_proportional\
			 income_discrimination_on_location_dependency\
			 income_discrimination_on_proportional
#			 sex_discrimination_on_location\
			 income_discrimination_on_location

GNUPLOT := gnuplot
EPS2PDF := epstopdf
PDFFILES = $(FIGS:=.pdf)
OUTPUT_TYPE := post
OUTPUT_EXT := eps

all: $(PDFFILES)
	@echo "[DONE]"

%.pdf: %.eps
	@$(EPS2PDF) $<

%.eps: %.plot %.csv
	@echo "[PLOT] $(*)"
	@$(GNUPLOT) -e "set output \"$(*).$(OUTPUT_EXT)\";" $<

clean:
	@rm -f $(PDFFILES)
