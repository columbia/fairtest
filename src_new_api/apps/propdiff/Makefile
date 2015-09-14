all: propdiff

propdiff: propdiff.o binomial.o
	$(CXX) -Wall -O3 -o $@ $+ 

%.o: %.cc %.h Makefile
	$(CXX) -Wall -O3 -c $<

%.o: %.cc Makefile
	$(CXX) -Wall -O3 -c $<

test: propdiff
	./propdiff 0.05 138 0 13 6
	./propdiff 0.05 27408 592 13701 2070

clean:
	$(RM) *.o propdiff
