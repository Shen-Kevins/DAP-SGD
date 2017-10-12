#ifndef _NUCLEARDATAPOINT_
#define _NUCLEARDATAPOINT_

#include "./Datapoint.h"

class NuclearDatapoint : public Datapoint{
public:
    NuclearDatapoint(const std::string &data_dir){
        this->feature.load(data_dir+"nuclear_feature_mat");;
        this->label.load(data_dir+"nuclear_label_mat");
    }
    
};
#endif