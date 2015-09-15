.PHONY: clean
INPUT_FOLDER=/home/vatlidak/repos/fairtest/data/
OUTPUT_FOLDER=.

all: adult berkeley recommender staples benchmark

adult:
	./make_adult.py $(INPUT_FOLDER)/adult/adult.csv > $(OUTPUT_FOLDER)/results_adult.txt

berkeley:
	./make_berkeley.py $(INPUT_FOLDER)/berkeley/berkeley.csv > $(OUTPUT_FOLDER)/results_berkeley.txt

recommender:
	./make_recommender.py $(INPUT_FOLDER)/recommender/recommendations.csv > $(OUTPUT_FOLDER)/results_recommender.txt

benchmark:
	./make_benchmark.py $(INPUT_FOLDER)/benchmark/benchmark.csv > $(OUTPUT_FOLDER)/results_benchmark.txt

staples:
	./make_staples.py $(INPUT_FOLDER)/staples/staples.csv > $(OUTPUT_FOLDER)/results_staples.txt

clean:
	rm -f $(OUTPUT_FOLDER)/results_recommender.txt
	rm -f $(OUTPUT_FOLDER)/results_benchmark.txt
	rm -f $(OUTPUT_FOLDER)/results_staples.txt
	rm -f $(OUTPUT_FOLDER)/results_berkeley.txt
	rm -f $(OUTPUT_FOLDER)/results_adult.txt
