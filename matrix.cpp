#include "matrix.h"
#include <stdexcept>
#include <fstream>
#include <cmath>


// Implementation of the constructor
Mat::Mat(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

// Implementation of the multiply function
Mat Mat::multiply(const Mat& a, const Mat& b) {
    // Basic validation could be added here: if (a.cols != b.rows) ...
    
    Mat result(a.rows, b.cols);

    #pragma omp parallel for
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            for (int k = 0; k < a.cols; k++) {
                result.data[i * b.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
        }
    }
    return result;
}


// Mat Mat::multiply(const Mat& a, const Mat& b) {
//     Mat result(a.rows, b.cols);

//     #pragma omp parallel for
//     for (int i = 0; i < a.rows; i++) {
//         for (int k = 0; k < a.cols; k++) {
//             // We pull this value out once so the inner loop is faster
//             double scalar = a.data[i * a.cols + k];
            
//             for (int j = 0; j < b.cols; j++) {
//                 // Now, 'j' is the inner-most index. 
//                 // b.data and result.data are accessed in a perfect straight line!
//                 result.data[i * b.cols + j] += scalar * b.data[k * b.cols + j];
//             }
//         }
//     }
//     return result;
// }



Mat Mat::add(const Mat& a, const Mat& b) {
    Mat result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            // Supports "Broadcasting": adds a single row bias to every row of A
            int b_idx = (b.rows == 1) ? j : (i * b.cols + j);
            result.data[i * a.cols + j] = a.data[i * a.cols + j] + b.data[b_idx];
        }
    }
    return result;
}

Mat Mat::apply_relu(const Mat& m) {
    Mat result = m;
    for(double &val : result.data) {
        val = (val > 0) ? val : 0;
    }
    return result;
}


inline double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}


Mat Mat::apply_swish(const Mat& m) {
    Mat result = m;
    int size = (int)result.data.size();
    for(double &val : result.data) {
        val = val * sigmoid(val);
    }
    
    // #pragma omp parallel for
    // for(int i = 0; i < size; i++) 
    // {
    //     result.data[i] = result.data[i] * sigmoid(result.data[i]);
    // }
    return result;
}

Mat Mat::swish_derivative(const Mat& input_z) {
    Mat grad(input_z.rows, input_z.cols);
    for (int i = 0; i < input_z.data.size(); i++) {
        double z = input_z.data[i];
        double s = sigmoid(z);
        double fz = z * s; // This is the swish(z) value
        
        // Swish derivative: f(z) + sigmoid(z) * (1 - f(z))
        grad.data[i] = fz + s * (1.0 - fz);
    }
    return grad;
}


Mat Mat::relu_derivative(const Mat& activated_matrix) {
    Mat grad = activated_matrix;
    for(double &val : grad.data) {
        // If the output was > 0, the slope is 1. Otherwise, it's 0.
        val = (val > 0) ? 1.0 : 0.0;
    }
    return grad;
}

Mat Mat::multiply_elements(const Mat& a, const Mat& b) {
    Mat result(a.rows, a.cols);
    for (int i = 0; i < a.data.size(); i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}

 Mat Mat::subtract(const Mat& a, const Mat& b) {
    // Safety check: ensure shapes match
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::runtime_error("Mat dimensions must match for subtraction.");
    }
    
    Mat result(a.rows, a.cols);
    for (int i = 0; i < a.data.size(); i++) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

Mat Mat::transpose(const Mat& m) {
    Mat result(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            // result[j][i] = m[i][j]
            result.data[j * m.rows + i] = m.data[i * m.cols + j];
        }
    }
    return result;
}

void Mat::save(std::ofstream& out) {
    out.write((char*)&rows, sizeof(int));
    out.write((char*)&cols, sizeof(int));
    out.write((char*)data.data(), data.size() * sizeof(double));
}

void Mat::load(std::ifstream& in) {
    in.read((char*)&rows, sizeof(int));
    in.read((char*)&cols, sizeof(int));
    data.resize(rows * cols);
    in.read((char*)data.data(), data.size() * sizeof(double));
}