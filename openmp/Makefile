# Makefile for Dijkstra OpenMP Program with .c file using C++ features

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = tag
SRC = openmp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
