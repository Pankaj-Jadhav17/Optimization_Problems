CC     = gcc
CFLAGS = -Wall -std=c11
TARGET = MS2509

all: $(TARGET)

$(TARGET): MS2509_main.c MS2509_fun.c MS2509_param.h MS2509_fun.h
	$(CC) $(CFLAGS) MS2509_main.c MS2509_fun.c -o $(TARGET)
	@echo "Build OK -> ./$(TARGET)"

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run