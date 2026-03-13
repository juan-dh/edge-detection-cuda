# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 \
		   -I/usr/local/cuda/include \
		   -Iinclude \
		   -I/usr/local/cuda/samples/Common/UtilNPP
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lfreeimage

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib

TARGET = $(BIN_DIR)/edge_detector
SRC = $(SRC_DIR)/main.cu

# Define the default rule
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

run: $(TARGET)
	@$(TARGET)

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."
