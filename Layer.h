#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

class Layer {
public:
    Mat weights;
    Mat bias;
    Mat lastInput;
    Mat lastOutput;

    Layer(int in_features, int out_features);
    Mat forward(const Mat& input);
};

#endif