CXX = g++-8
CXXFLAGS = -std=c++17 -O3 -march=native -funroll-loops 

.PHONY: all
all: ising3d figures 

ising3d: ising3d.cpp

results: ising3d
	./ising3d 1000000 1 0 30 > results_30.dat
	./ising3d 1000000 1 0 60 > results_60.dat

figures: results figures.gplt
	gnuplot figures.gplt

.PHONY: clean
clean:
	@rm -rf task *.dat *.eps *.aux *.log *.synctex 