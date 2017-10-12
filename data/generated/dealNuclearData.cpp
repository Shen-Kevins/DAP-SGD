#include<iostream>
#include<armadillo>

using namespace arma;
using namespace std;

int main(){
    ifstream feature_in;
    ifstream weight_in;
    ifstream label_in;
    feature_in.open("nuclear_feature.txt");
    weight_in.open("nuclear_weight.txt");
    label_in.open("nuclear_label.txt");

    mat feature(4000,50);
    mat weight(50,40);
    mat label(4000,40);

    for(int i = 0; i < 4000; i++){
        for(int j = 0; j < 50; j++){
            feature_in >> feature(i,j);
        }
        for(int k = 0; k < 40; k++){
            label_in >> label(i,k);
        }
    }
    for(int i = 0; i < 50; i++){
        for(int j = 0;j < 40; j++){
            weight_in >> weight(i,j);
        }
    }
    feature.save("nuclear_feature_mat");
    weight.save("nuclear_weight_mat");
    label.save("nuclear_label_mat");

    return 0;
}
