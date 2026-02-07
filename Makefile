# Variables
CC = g++
CFLAGS = -I./include
LDFLAGS = -L./lib -static
LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm -static-libgcc -mwindows -O2
SRC = simulation.cpp nn.cpp
OBJ = game.exe

# The default target
all: $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(SRC) -o $(OBJ) $(CFLAGS) $(LDFLAGS) $(LIBS)

# Clean up the executable
clean:
	del $(OBJ)