#include "Layer.h"
#include <random>
#include "matrix.h"

Layer::Layer(int in_features, int out_features) 
    : weights(in_features, out_features), 
      bias(1, out_features), 
      lastInput(0,0), 
      lastOutput(0,0) {
    
    // Random initialization (Xavier/Glorot light)
    // std::default_random_engine generator;
    static std::default_random_engine generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-0.1, 0.1);

    for (double &val : weights.data) {
        val = distribution(generator);
    }
    for (double &val : bias.data) {
        val = 0.0; // Biases are usually safe to start at 0
    }
}

Mat Layer::forward(const Mat& input) {
    lastInput = input;
    // Y = (X * W) + B
    lastOutput = Mat::add(Mat::multiply(input, weights), bias);
    return lastOutput;
}