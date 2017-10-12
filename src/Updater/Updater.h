#ifndef _UPDATER_
#define _UPDATER_

#include "../Model/Model.h"
#include "../Utils/Utils.h"

class Updater {
protected:
    Model *model;
    Datapoint *datapoint;

public:
    Updater(){}

    Updater(Model *model, Datapoint *datapoint){
        this->model = model;
        this->datapoint = datapoint;
    }

    void ApplyGradient(){
        // mat gradient = least_backward(datapoint->GetFeature(),vec2mat(model->GetWeight(),datapoint->GetFeature().n_cols,datapoint->GetLabel().n_cols),datapoint->GetLabel());
        // vector<double> gradients = mat2vec(gradient);
        // // gradients += L2_lambda * model->GetWeight();
        // for(int i = 0; i < model->GetSize(); i++){
        //     gradients[i] += L2_lambda * model->GetWeight()[i];
        // }

        // model->UpdateWeight(learning_rate,gradients);
    }
    
    virtual vector<double> ApplySGD()=0;


    ~Updater(){}
};

#endif