.PHONY: clean
INPUT_FOLDER=../../../data
OUTPUT_FOLDER=results

export PYTHONPATH=../../../src

all: adult berkeley recommender staples medical_bin medical_reg overfeat csv

adult:
	rm -r $(OUTPUT_FOLDER)/adult; \
	mkdir $(OUTPUT_FOLDER)/adult; \
	python -m make_adult $(INPUT_FOLDER)/adult/adult.csv $(OUTPUT_FOLDER)/adult > $(OUTPUT_FOLDER)/results_adult.csv

benchmark:
	rm -r $(OUTPUT_FOLDER)/benchmark; \
	mkdir $(OUTPUT_FOLDER)/benchmark; \
	python -m make_becnhmark $(INPUT_FOLDER)/benchmark/benchmark.csv $(OUTPUT_FOLDER)/benchmark > $(OUTPUT_FOLDER)/results_benchmark.csv

berkeley:
	rm -r $(OUTPUT_FOLDER)/berkeley; \
	mkdir $(OUTPUT_FOLDER)/berkeley; \
	python -m make_berkeley $(INPUT_FOLDER)/berkeley/berkeley.csv $(OUTPUT_FOLDER)/berkeley > $(OUTPUT_FOLDER)/results_berkeley.csv

recommender:
	rm -rf $(OUTPUT_FOLDER)/recommender; \
	mkdir $(OUTPUT_FOLDER)/recommender; \
	mkdir $(OUTPUT_FOLDER)/recommender/test; \
	mkdir $(OUTPUT_FOLDER)/recommender/error; \
	mkdir $(OUTPUT_FOLDER)/recommender/discovery; \
	python -m make_recommender $(INPUT_FOLDER)/recommender/recommendations.txt $(OUTPUT_FOLDER)/recommender > $(OUTPUT_FOLDER)/results_recommender.csv

staples:
	rm -r $(OUTPUT_FOLDER)/staples; \
	mkdir $(OUTPUT_FOLDER)/staples; \
	python -m make_staples $(INPUT_FOLDER)/staples/staples.csv $(OUTPUT_FOLDER)/staples > $(OUTPUT_FOLDER)/results_staples.csv

medical_bin:
	rm -r $(OUTPUT_FOLDER)/medical_bin; \
	mkdir $(OUTPUT_FOLDER)/medical_bin; \
	python -m make_medical_bin $(INPUT_FOLDER)/medical/predictions_bin.csv  $(OUTPUT_FOLDER)/medical_bin > $(OUTPUT_FOLDER)/results_medical_bin.csv
	
medical_reg:
	rm -r $(OUTPUT_FOLDER)/medical_reg; \
	mkdir $(OUTPUT_FOLDER)/medical_reg; \
	python -m make_medical_reg $(INPUT_FOLDER)/medical/predictions_reg_log.csv $(OUTPUT_FOLDER)/medical_reg > $(OUTPUT_FOLDER)/results_medical_reg.csv

overfeat:
	rm -r $(OUTPUT_FOLDER)/overfeat; \
	mkdir $(OUTPUT_FOLDER)/overfeat; \
	python -m make_overfeat $(INPUT_FOLDER)/images/overfeat_raw.txt  $(OUTPUT_FOLDER)/overfeat > $(OUTPUT_FOLDER)/results_overfeat.csv

csv:
	./make_csv.sh results/results $(OUTPUT_FOLDER)/timing_percentages.csv

benchmarks: benchmark_performance benchmark_effectiveness

benchmark_effectiveness:
	python -m make_benchmark_effectiveness $(INPUT_FOLDER)/staples/staples.csv 1 >  $(OUTPUT_FOLDER)/benchmark_effectiveness.csv

benchmark_performance:
	python -m make_benchmark_performance $(INPUT_FOLDER)/staples/staples.csv 2 > $(OUTPUT_FOLDER)/benchmark_performance.csv

clean:
	rm -rf $(OUTPUT_FOLDER)/*
