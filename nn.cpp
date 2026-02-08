const double DIST_SCALE = 1.5e11;
const double VEL_SCALE = 30000.0;

#include <vector>
#include "nn.h"
#include <cstdlib>

using namespace std;

vector<double> normalize_input(double x, double y, double vx, double vy)
{
    return {x / DIST_SCALE, y / DIST_SCALE, vx / VEL_SCALE, vy / VEL_SCALE};
}

vector<double> normalize_target(double x, double y)
{
    return {x / DIST_SCALE, y / DIST_SCALE};
}

DenseLayer::DenseLayer(int in, int out) : in_nodes(in), out_nodes(out)
{
    weights.resize(in, vector<double>(out));
    // populating the weights with random weights. DONOT initalize with zero as all neurons would work the same way and break symmetry.
    biases.resize(out, 0.0);
    for (int i = 0; i < in; i++)
    {
        for (int j = 0; j < out; j++)
            weights[i][j] = ((double)rand() / RAND_MAX) * 0.2 - 0.1; // Small random weights
    }
}

std::vector<double> DenseLayer::forward(const vector<double> &input)
{
    last_input = input;
    vector<double> output(out_nodes,0.0);
    for(int j=0;j<out_nodes;j++)
    {
        for(int i=0;i<in_nodes;i++)
        {
            output[j] += input[i]*weights[i][j];
        }
        output[j]  += biases[j];
        output[j] = tanh(output[j]);
    }
    last_output = output;
    return output;
}


void train(DenseLayer &layer, vector<double> target, double learningRate)
{
    vector <double> error(layer.out_nodes);
    for(int i=0;i<layer.out_nodes;i++)
    {
        double delta = (layer.last_output[i]-target[i]) * (1-layer.last_output[i]*layer.last_output[i]);  //error - derivative of tanh(which was the output)
        error[i] = delta;
    }

    // update the weights
    for(int j=0;j<layer.out_nodes;j++)
    {
        for(int i=0;i<layer.in_nodes;i++)
        {
            layer.weights[i][j] -= learningRate*error[j]*layer.last_input[i];
        }
        layer.biases[j]-= learningRate*error[j];
    }
}

