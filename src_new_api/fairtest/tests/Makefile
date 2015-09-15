.PHONY: clean

all: adult berkeley recommender staples benchmark

adult:
	./make_adult.py /home/vatlidak/repos/fairtest/data/adult/adult.csv > results_adult.txt

berkeley:
	./make_berkeley.py /home/vatlidak/repos/fairtest/data/berkeley/berkeley.csv > results_berkeley.txt

recommender:
	./make_recommender.py /home/vatlidak/repos/fairtest/data/recommender/recommendations.csv > results_recommender.txt

benchmark:
	./make_benchmark.py /home/vatlidak/repos/fairtest/data/benchmark/benchmark.csv > results_benchmark.txt

staples:
	./make_staples.py /home/vatlidak/repos/fairtest/data/staples/staples.csv > results_staples.txt

clean:
	rm -f results_recommender.txt
	rm -f results_benchmark.txt
	rm -f results_staples.txt
	rm -f results_berkeley.txt
	rm -f results_adult.txt
