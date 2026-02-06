# Variables
CC = g++
CFLAGS = -I./include
LDFLAGS = -L./lib
LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm
SRC = simulation.cpp nn.cpp
OBJ = game.exe

# The default target
all: $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(SRC) -o $(OBJ) $(CFLAGS) $(LDFLAGS) $(LIBS)

# Clean up the executable
clean:
	del $(OBJ)