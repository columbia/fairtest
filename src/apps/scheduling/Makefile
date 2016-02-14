.PHONY: clean

UNITS=1000

INPUT_FOLDER=./data
OUTPUT_FOLDER=./results
SCRIPTS=./scripts
JOIN=$(SCRIPTS)/join.py
SCHEDULE=./scheduling.py

all: shifts

shifts:
	@mkdir -p $(OUTPUT_FOLDER)
	$(SCHEDULE) $(INPUT_FOLDER)/dallas_police_reports.json $(INPUT_FOLDER)/demographics.txt $(UNITS)
	@mv shifts.txt $(OUTPUT_FOLDER)

clean:
	rm -rf $(OUTPUT_FOLDER)/*
	rmdir $(OUTPUT_FOLDER)
	@rm -rf ./shifts.txt


