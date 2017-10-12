#include<iostream>
#include<armadillo>

using namespace arma;
using namespace std;

int main(){
    ifstream feature_in;
    ifstream weight_in;
    ifstream label_in;
    feature_in.open("feature.txt");
    weight_in.open("weight.txt");
    label_in.open("label.txt");

    mat feature(1000,5000);
    mat weight(5000,1);
    mat label(1000,1);

    for(int i = 0; i < 1000; i++){
        label_in >> label(i,0);
        for(int j = 0; j < 5000; j++){
            feature_in >> feature(i,j);
            weight_in >> weight(j,0);
        }
    }
    feature.save("feature_mat");
    weight.save("weight_mat");
    label.save("label_mat");

    return 0;
}
