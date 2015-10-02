.PHONY: clean
INPUT_FOLDER=/home/vatlidak/repos/fairtest/data
OUTPUT_FOLDER=.

all: adult berkeley recommender recommender_reg staples medical_bin medical_reg overfeat csv

adult:
	./make_adult.py $(INPUT_FOLDER)/adult/adult.csv > $(OUTPUT_FOLDER)/results_adult.txt

berkeley:
	./make_berkeley.py $(INPUT_FOLDER)/berkeley/berkeley.csv > $(OUTPUT_FOLDER)/results_berkeley.txt

recommender:
	./make_recommender.py $(INPUT_FOLDER)/recommender/recommendations_labels.txt > $(OUTPUT_FOLDER)/results_recommender_labels.txt

recommender_reg:
	./make_recommender_reg.py $(INPUT_FOLDER)/recommender/recommendations_types.txt > $(OUTPUT_FOLDER)/results_recommender_types.txt

staples:
	./make_staples.py $(INPUT_FOLDER)/staples/staples.csv > $(OUTPUT_FOLDER)/results_staples.txt

medical_bin:
	./make_medical_bin.py $(INPUT_FOLDER)/medical/predictions_bin_4_25.csv >  $(OUTPUT_FOLDER)/results_medical_bin.txt
	
medical_reg:
	./make_medical_reg.py $(INPUT_FOLDER)/medical/predictions_reg_25.csv >  $(OUTPUT_FOLDER)/results_medical_reg.txt

overfeat:
	./make_overfeat.py $(INPUT_FOLDER)/images/overfeat_raw.txt >  $(OUTPUT_FOLDER)/results_overfeat.txt

csv:
	./make_csv.sh results

benchmarks: benchmark_performance benchmark_effectiveness

benchmark_effectiveness:
	./make_benchmark_effectiveness.py $(INPUT_FOLDER)/staples/staples.csv 1 >  $(OUTPUT_FOLDER)/results_benchmark_effectiveness.csv

benchmark_performance:
	./make_benchmark_performance.py $(INPUT_FOLDER)/staples/staples.csv 2 > $(OUTPUT_FOLDER)/results_benchmark_performance.csv

clean:
	rm -f $(OUTPUT_FOLDER)/results_*.txt
	rm -f $(OUTPUT_FOLDER)/*.csv
