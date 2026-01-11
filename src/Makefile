CXX = g++ -std=c++17
CFLAGS = -O2
CV_FLAGS = `pkg-config --cflags --libs opencv4`
SRC = infer_onnx.cpp
TARGET = infer_yolo

all: clean build run

build:
	@echo "🔨 Компиляция..."
	$(CXX) $(CFLAGS) $(SRC) -o $(TARGET) $(CV_FLAGS)

run:
	@echo ""
	./$(TARGET)

clean:
	@echo "🧹 Очистка..."
	rm -rf $(TARGET) cpp_results

.PHONY: all build run clean
