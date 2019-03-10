CXX = g++-8
CXXFLAGS = -std=c++17 -O3 -march=native -funroll-loops 

.PHONY: all
all: ising3d figures 

ising3d: ising3d.cpp

results: ising3d
	./ising3d 100 1 0 10 > results_10.dat
	./ising3d 100 1 0 20 > results_20.dat

figures: results figures.gplt
	gnuplot figures.gplt

.PHONY: clean
clean:
	@rm -rf task *.dat *.eps *.aux *.log *.synctex 