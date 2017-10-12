#ifndef _L2L1MODEL_
#define _L2L1MODEL_

#include "./Model.h"

class L2L1Model : public Model{
public:
    L2L1Model(int length) : Model(length){}

    virtual double ComputeRegularization() override{
        double regloss = 0;
        for(int i = 0; i < size; i++){
            regloss += std::abs(weight[i]);
        }
        return L1_lambda * regloss;
    }

    double ComputeLoss(Datapoint *datapoint) override{
        double loss = 0;
        if(this->size == 784){
            loss = logistic_forward(datapoint->GetFeature(),vec2mat(this->weight,size,1),datapoint->GetLabel());
        }
        else{
            // loss = accu(square(datapoint->GetFeature() * weight-(datapoint->GetLabel()))) /(2 * datapoint->GetSize());
            loss = least_forward(datapoint->GetFeature(),vec2mat(this->weight,size,1),datapoint->GetLabel());
        }
        return loss + ComputeL2Loss() + ComputeRegularization();
    }

    void ProxyOperator(vector<double> &local_weight, double gama) override{
        for(int i = 0; i < local_weight.size(); i ++){
            double val = local_weight[i];
            double sign = val > 0? 1:-1;
            local_weight[i] = sign * fmax(std::abs(val)-gama, 0);
        }
    }    

};

#endif