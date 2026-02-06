#pragma once
using namespace std;

#include <vector>
#include <cmath>

struct DenseLayer {
    int in_nodes, out_nodes;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> last_input;
    std::vector<double> last_output;

    DenseLayer(int in, int out); // Declaration
    std::vector<double> forward(const std::vector<double>& input); // Declaration
};

vector<double> normalize_target(double x, double y);

vector<double> normalize_input(double x, double y, double vx, double vy);

void train(DenseLayer &layer, vector<double> target, double learningRate);