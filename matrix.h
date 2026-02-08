#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <fstream>


struct Mat {
    int rows, cols;
    std::vector<double> data;

    // Constructor
    Mat(int r, int c);
    
    // The most important function in Deep Learning
    static Mat multiply(const Mat& a, const Mat& b);
    static Mat add(const Mat& a, const Mat& b);
    static Mat apply_relu(const Mat& m);
    static Mat relu_derivative(const Mat& activated_matrix);
    static Mat multiply_elements(const Mat& a, const Mat& b);
    static  Mat subtract(const Mat& a, const Mat& b) ;
    static Mat transpose(const Mat& m);
    void save(std::ofstream& out);
    void load(std::ifstream& in);
};

#endif // MATRIX_H