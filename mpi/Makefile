CXX = mpicxx
CXXFLAGS = -Wall -O2 -std=c++17
TARGET = tag
SRCS = mpi.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
