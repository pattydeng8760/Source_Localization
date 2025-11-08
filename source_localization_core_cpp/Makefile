# Compiler (use hdf5's C++ compiler wrapper)
CXX = h5c++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall

# Executable name
TARGET = read_hdf5

# Source file
SRC = read_hdf5.cpp

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
