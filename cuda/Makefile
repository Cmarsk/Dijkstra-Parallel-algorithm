# 编译器和选项
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_52 

# 可执行文件名
TARGET = tag

# 源文件
SRC = cuda.cu

# 默认目标
all: $(TARGET)

# 构建规则
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# 清理
clean:
	rm -f $(TARGET)
