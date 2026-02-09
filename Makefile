# Variables
CC = g++
CFLAGS = -I./include
LDFLAGS = -L./lib 
LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm -static-libgcc -march=native -ffast-math -flto
# LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm -static-libgcc -mwindows -O2
SRC = simulation.cpp nn.cpp Layer.cpp matrix.cpp
OBJ = game.exe
Optionals = -fopenmp 
Production = -static-libcc -mwindows
Optimization = -O3

# The default target
all: $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(SRC) -o $(OBJ) $(CFLAGS) $(LDFLAGS) $(LIBS) 

# Clean up the executable
clean:
	del $(OBJ)