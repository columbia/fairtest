FIGS = race_discrimination_on_location_dependency\
       race_discrimination_on_proportional\
			 income_discrimination_on_location_dependency\
			 income_discrimination_on_proportional\
			 sex_discrimination_on_proportional\
			 sex_discrimination_on_location_dependency\
			 income_per_race

GNUPLOT := gnuplot
EPS2PDF := epstopdf
PDFFILES = $(FIGS:=.pdf)
OUTPUT_TYPE := post
OUTPUT_EXT := eps

.PHONY: misc all clean

all: misc $(PDFFILES)
	@echo "[DONE]"

misc:
	+make -C $@

%.pdf: %.eps
	@$(EPS2PDF) $<

%.eps: %.plot %.csv
	@echo "[PLOT] $(*)"
	@$(GNUPLOT) -e "set output \"$(*).$(OUTPUT_EXT)\";" $<

clean:
	@rm -f $(PDFFILES)
	+make -C misc clean
