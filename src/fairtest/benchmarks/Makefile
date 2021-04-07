.PHONY: clean
PYTHON=python2.7
INPUT_FOLDER=../../../data
OUTPUT_FOLDER=results

export $(PYTHON)PATH=../../../src

all: adult berkeley recommender staples medical images csv

adult:
	rm -r $(OUTPUT_FOLDER)/adult; \
	mkdir -p $(OUTPUT_FOLDER)/adult; \
	$(PYTHON) -m make_adult $(INPUT_FOLDER)/adult/adult.csv $(OUTPUT_FOLDER)/adult > $(OUTPUT_FOLDER)/results_adult.csv

benchmark:
	rm -r $(OUTPUT_FOLDER)/benchmark; \
	mkdir -p $(OUTPUT_FOLDER)/benchmark; \
	$(PYTHON) -m make_becnhmark $(INPUT_FOLDER)/benchmark/benchmark.csv $(OUTPUT_FOLDER)/benchmark > $(OUTPUT_FOLDER)/results_benchmark.csv

berkeley:
	rm -r $(OUTPUT_FOLDER)/berkeley; \
	mkdir -p $(OUTPUT_FOLDER)/berkeley; \
	$(PYTHON) -m make_berkeley $(INPUT_FOLDER)/berkeley/berkeley.csv $(OUTPUT_FOLDER)/berkeley > $(OUTPUT_FOLDER)/results_berkeley.csv

recommender:
	rm -rf $(OUTPUT_FOLDER)/recommender; \
	mkdir -p $(OUTPUT_FOLDER)/recommender; \
	mkdir $(OUTPUT_FOLDER)/recommender/test; \
	mkdir $(OUTPUT_FOLDER)/recommender/error; \
	mkdir $(OUTPUT_FOLDER)/recommender/discovery; \
	$(PYTHON) -m make_recommender $(INPUT_FOLDER)/recommender/recommendations.txt $(OUTPUT_FOLDER)/recommender > $(OUTPUT_FOLDER)/results_recommender.csv

staples:
	rm -r $(OUTPUT_FOLDER)/staples; \
	mkdir -p $(OUTPUT_FOLDER)/staples; \
	$(PYTHON) -m make_staples $(INPUT_FOLDER)/staples/staples.csv $(OUTPUT_FOLDER)/staples > $(OUTPUT_FOLDER)/results_staples.csv

medical:
	rm -r $(OUTPUT_FOLDER)/medical; \
	mkdir -p $(OUTPUT_FOLDER)/medical; \
	$(PYTHON) -m make_medical $(INPUT_FOLDER)/medical/predictions_reg_log.csv $(OUTPUT_FOLDER)/medical > $(OUTPUT_FOLDER)/results_medical.csv

images:
	rm -r $(OUTPUT_FOLDER)/images; \
	mkdir -p $(OUTPUT_FOLDER)/images; \
	$(PYTHON) -m make_images $(INPUT_FOLDER)/images/caffe.txt  $(OUTPUT_FOLDER)/images > $(OUTPUT_FOLDER)/results_images.csv

scheduling:
	rm -r $(OUTPUT_FOLDER)/scheduling; \
	mkdir -p $(OUTPUT_FOLDER)/scheduling; \
	$(PYTHON) -m make_scheduling $(INPUT_FOLDER)/scheduling/shifts.txt  $(OUTPUT_FOLDER)/scheduling > $(OUTPUT_FOLDER)/results_scheduling.csv



csv:
	./make_csv.sh results/results $(OUTPUT_FOLDER)/timing_percentages.csv

benchmarks: benchmark_performance benchmark_effectiveness

benchmark_effectiveness:
	$(PYTHON) -m make_benchmark_effectiveness $(INPUT_FOLDER)/staples/staples.csv 10 > $(OUTPUT_FOLDER)/benchmark_effectiveness.csv

benchmark_performance:
	$(PYTHON) -m make_benchmark_performance $(INPUT_FOLDER)/staples/staples.csv 10 > $(OUTPUT_FOLDER)/benchmark_performance.csv

clean:
	rm -rf $(OUTPUT_FOLDER)/*
