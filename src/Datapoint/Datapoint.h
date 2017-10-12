#ifndef _DATAPOINT_
#define _DATAPOINT_

#include<armadillo>
#include<string.h>

using namespace arma;

class Datapoint {
protected:
    mat feature;
    mat label;
public:
    Datapoint(){}
    
    Datapoint(const std::string &data_dir){
        this->feature.load(data_dir+"feature_mat");;
        this->label.load(data_dir+"label_mat");
        //  std::cout<< feature.n_cols << feature.n_rows;
    }

    mat GetFeaturesRows(int left, int right){
        return this->feature.rows(left, right);
    }

    mat GetLabelsRols(int left, int right){
        return this->label.rows(left, right);
    }

    int GetSize(){
        return label.n_rows;
    }

    mat GetFeature(){
        return this->feature;
    }

    mat GetLabel(){
        return this->label;
    }
};

#endif