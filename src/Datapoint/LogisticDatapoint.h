#ifndef _LOGISTICDATAPOINT_
#define _LOGISTICDATAPOINT_

#include "./Datapoint.h"

class LogisticDatapoint : public Datapoint{
public:
    LogisticDatapoint(const std::string &data_dir){
        this->feature.load(data_dir+"mnist_train_feature_mat01");
        this->label.load(data_dir+"mnist_train_label_mat01");

        this->feature = this->feature / 255.001;
    }
    
};
#endif