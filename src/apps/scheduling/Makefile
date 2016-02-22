.PHONY: clean

UNITS=2000

INPUT_FOLDER=./data
OUTPUT_FOLDER=../../../data/scheduling
SCHEDULE=./scheduling.py

all: shifts

shifts:
	$(SCHEDULE) $(INPUT_FOLDER)/dallas_police_reports.json $(INPUT_FOLDER)/demographics.txt $(UNITS)
	@mkdir -p $(OUTPUT_FOLDER)
	mv shifts.txt $(OUTPUT_FOLDER)

clean:
	rm -rf $(OUTPUT_FOLDER)/*
	rmdir $(OUTPUT_FOLDER)
	@rm -rf ./shifts.txt


