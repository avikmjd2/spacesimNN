#include <raylib.h>
#include <bits/stdc++.h>
#include <fstream>
#include "nn.h"
#include <algorithm>
#include <random>

const double DIST_SCALE = 1.5e11;
const double VEL_SCALE = 30000.0;

using namespace std;

// struct Vector2 {
//     double x,y;
// };
ofstream dataFile("trainingData.csv");

enum State
{
    COLLECTING,
    TRAINING,
    PREDICTING
};

class Body
{
public:
    Vector2 pos;
    Vector2 vel;
    double mass;

    Body(float x, float y, float vx, float vy, double m)
    {
        pos = {x, y};
        vel = {vx, vy};
        mass = m;
    }
    void update_position(double dt)
    {
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
    }
};

void apply_gravity(Body &a, Body &b, double dt)
{
    double G = 6.674e-11;
    double dx = b.pos.x - a.pos.x;
    double dy = b.pos.y - a.pos.y;
    double distance = sqrt(dx * dx + dy * dy);
    double force = (G * a.mass * b.mass) / (distance * distance + 0.001);
    double ax = force * (dx / distance) / a.mass;
    double ay = force * (dy / distance) / a.mass;

    a.vel.x += ax * dt;
    a.vel.y += ay * dt;
}

void apply_physics(std::vector<Body> &bodies, double dt)
{
    for (auto i = 0; i < bodies.size(); i++)
    {
        for (auto j = 0; j < bodies.size(); j++)
        {
            if (i == j)
                continue;
            ;

            apply_gravity(bodies[i], bodies[j], dt);
        }
    }

    // 2. Update Positions
    for (auto &b : bodies)
    {
        b.update_position(dt);
    }
}

void run_training(DenseLayer &nn, int epochs, double lr)
{
    // 1. Load the data into memory once
    vector<string> allLines;
    ifstream file("trainingData.csv");
    string line;
    while (getline(file, line)) {
        if (!line.empty()) allLines.push_back(line);
    }
    file.close();

    // Setup the random engine for shuffling
    std::default_random_engine rng(std::random_device{}());

    for (int e = 0; e < epochs; e++)
    {
        // 2. Shuffle the lines at the start of every epoch
        std::shuffle(allLines.begin(), allLines.end(), rng);

        double totalError = 0;
        for (const string& rowLine : allLines)
        {
            stringstream ss(rowLine);
            string val;
            vector<double> row;

            while (getline(ss, val, ',')) {
                row.push_back(stod(val));
            }

            if (row.size() == 8)
            {
                vector<double> input = normalize_input(row[0], row[1], row[2], row[3]);
                vector<double> target = normalize_target(row[4], row[5]);

                vector<double> prediction = nn.forward(input);
                train(nn, target, lr);

                totalError += pow(prediction[0] - target[0], 2) + pow(prediction[1] - target[1], 2);
            }
        }
        
        if (e % 10 == 0)
            cout << "Epoch " << e << " | Average Loss: " << totalError / allLines.size() << endl;
    }
}

int frameCount = 0;
int FRAME = 4000;
int main()
{
    const int screenwidth = 800;
    const int screenheight = 800;
    InitWindow(screenwidth, screenheight, "Simulation");
    SetTargetFPS(60);

    double scale = 1e9;
    vector<Body> bodies;
    double dt = 3600 * 5;
    // Body sun(0, 0, 0, 0, 1.989e30);
    bodies.emplace_back(0, 0, 0, 0, 1.989e30);          // sun
    bodies.emplace_back(1.5e11, 0, 0, 29780, 5.972e24); // earth
    // bodies.emplace_back(0, 1.5e11, -20000, 0, 5.972e25); //another planet
    // bodies.emplace_back(0.5e11, 1e11, -20000, 0, 10.972e25); //another planet
    // Body earth(1.5e11, 0, 0, 29780, 5.972e24);

    DenseLayer brain(4, 2);
    Vector2 ghostPos = {1.5e11, 0};
    State currentState = COLLECTING;

    while (!WindowShouldClose())
    {
        // input state
        double currentRelX = bodies[1].pos.x - bodies[0].pos.x;
        double currentRelY = bodies[1].pos.y - bodies[0].pos.y;
        double velX = bodies[1].vel.x;
        double velY = bodies[1].vel.y;

        apply_physics(bodies, dt);
        // earth.update_position(3600 * 2);
        BeginDrawing();
        ClearBackground(BLACK);
        // DrawCircle(screenwidth/2,screenheight/2,10,YELLOW);

        // output state
        if (frameCount < FRAME)
        {
            dataFile << currentRelX << "," << currentRelY << "," << velX << "," << velY << ",";
            // dataFile << bodies[1].pos.x << "," << bodies[1].pos.y << "," << bodies[1].vel.x << "," << bodies[1].vel.y << "\n";
            dataFile << bodies[1].pos.x - bodies[0].pos.x << ","
                     << bodies[1].pos.y - bodies[0].pos.y << ","
                     << bodies[1].vel.x << "," << bodies[1].vel.y << "\n"; // using relative psoition for portability
            frameCount++;
        }
        // else
        else if (frameCount == FRAME)
        {
            dataFile.close();
            cout << "Data collected" << endl;
            frameCount++;
            currentState = TRAINING;
        }
        if (currentState == TRAINING)
        {
            run_training(brain, 300, 0.005);
            currentState = PREDICTING;
        }
        else if (currentState == PREDICTING)
        {
            apply_physics(bodies, dt);
            // vector<double> input = normalize_input(bodies[1].pos.x, bodies[1].pos.y, bodies[1].vel.x, bodies[1].vel.y);
            vector<double> input = normalize_input(bodies[1].pos.x - bodies[0].pos.x,
                                                   bodies[1].pos.y - bodies[0].pos.y,
                                                   bodies[1].vel.x, bodies[1].vel.y);

            vector<double> output = brain.forward(input);
            double predictedRelX = output[0] * DIST_SCALE;
            double predictedRelY = output[1] * DIST_SCALE;

            // 2. Convert to Absolute (Relative -> Window Coordinates)
            // Add the Sun's current position so the ghost stays with the sun
            ghostPos.x = (float)(predictedRelX + bodies[0].pos.x);
            ghostPos.y = (float)(predictedRelY + bodies[0].pos.y);
        }

        for (size_t i = 0; i < bodies.size(); i++)
        {
            int drawX = (int)(bodies[i].pos.x / scale) + screenwidth / 2;
            int drawY = (int)(bodies[i].pos.y / scale) + screenheight / 2;

            // Color coding the bodies
            Color c = (i == 0) ? YELLOW : (i == 1) ? BLUE
                                                   : RED;
            DrawCircle(drawX, drawY, (i == 0 ? 10 : 5), c);
        }

        if (currentState == PREDICTING)
        {
            // Draw AI Ghost (RED)
            int gx = (ghostPos.x / scale) + 400;
            int gy = (ghostPos.y / scale) + 400;
            DrawCircle(gx, gy, 3, RED);
            DrawText("RED: AI PREDICTION", 10, 40, 20, RED);
            DrawText("BLUE: REAL PHYSICS", 10, 70, 20, BLUE);
        }

        // int drawX = (earth.pos.x / scale) + screenwidth/2;
        // int drawY = (earth.pos.y / scale) + screenheight/2;
        // DrawCircle(drawX, drawY, 4, BLUE);

        DrawText("Three Body", 10, 10, 20, RAYWHITE);

        EndDrawing();
    }
    CloseWindow();
}