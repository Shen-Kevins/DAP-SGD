#ifndef _UTILS_
#define _UTILS_

#include<armadillo>

using namespace arma;

double least_forward(const mat &feature, const mat &weight, const mat &label){
    return accu(square(feature * weight - label)) / (2.0 * feature.n_rows);
}

mat least_backward(const mat &feature, const mat &weight, const mat label){
    //mat gradient(weight.n_rows, weight.n_cols, fill::zeros);
    return feature.t() *(feature *weight - label)/feature.n_rows;
    //return gradient;
}

double logistic_forward(const mat &feature, const mat &weight, const mat &label){
    double a = accu(label.t() * log(pow(1 + exp(- feature * weight), -1))) / feature.n_rows;
    double b = accu((1 - label.t()) * log(1- pow(1 + exp(- feature * weight), -1))) / feature.n_rows;
    return -(a + b);
}

mat logistic_backward(const mat &feature, const mat &weight, const mat &label){
    return -feature.t() * (label - pow( 1 + exp(- feature * weight), -1)) / (feature.n_rows);
}

// mat sigmoid(){

// }


template<typename T>
T max_element(std::vector<T> vec){
    T max_val = vec[0];
    for(int i = 1; i < vec.size();i++){
        if(max_val < vec[i])
            max_val = vec[i];
    }
    return max_val;
}

mat vec2mat(const std::vector<double> &w, int row, int col){
    mat out(row,col);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            out(i,j) = w[i*col+j];
        }
    }
    return out;
}

std::vector<double> mat2vec(const mat &matrix){
    int row = matrix.n_rows;
    int col = matrix.n_cols;
    std::vector<double> out(row*col,0);
    for(int i = 0;i < row; i++){
        for(int j = 0;j < col; j++){
            out[i*col+j] = matrix(i,j);
        }
    }
    return out;
}


#endif