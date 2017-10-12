#ifndef _L2GROUPLASSOMODEL_
#define _L2GROUPLASSOMODEL_

#include "./Model.h"

class L2GroupLassoModel : public Model{
private:
    int group_size = 10;

public:

    L2GroupLassoModel(int length) : Model(length){}

    double ComputeRegularization() override{

        int group_num = this->size / group_size +(this->size % group_size != 0);

        double regloss = 0.0;
        for(int i = 0 ; i < group_num; i++){
            double loss_tp = 0;
            for(int j = 0; (j < group_size) && (i*group_size+j < this->size); j++ ){
                loss_tp += weight[i * group_size + j] * weight[i * group_size + j];
            }
            regloss += std::sqrt(loss_tp);
        }

        return regloss * GroupLasso_lambda;
    }
    
    double ComputeLoss(Datapoint *datapoint) override{
        double loss = 0;
        if(this->size == 784){
            loss = logistic_forward(datapoint->GetFeature(),vec2mat(this->weight,this->size,1),datapoint->GetLabel());
        }
        else{
            // loss = accu(square(datapoint->GetFeature() * weight-(datapoint->GetLabel()))) /(2 * datapoint->GetSize());
            loss = least_forward(datapoint->GetFeature(), vec2mat(this->weight,this->size,1),datapoint->GetLabel());
        }
        return loss + ComputeL2Loss() + ComputeRegularization();
    }
    
    void ProxyOperator(vector<double> &local_weight, double gama) override{
        int group_num = local_weight.size() / group_size +(local_weight.size() % group_size != 0);
        for(int i = 0; i < group_num; i++){
            double loss_tp = 0;
            for(int j = 0; (j < group_size) && (i*group_size+j < local_weight.size()); j++ ){
                loss_tp += local_weight[i * group_size + j] * local_weight[i * group_size + j];
            }
            loss_tp = std::sqrt(loss_tp);

            if(loss_tp <= gama){
                for(int j = 0; (j < group_size) && (i*group_size+j < local_weight.size()); j++ ){
                    local_weight[i * group_size + j] = 0.0;
                }
            }else{
                loss_tp = 1 - gama/loss_tp;
                for(int j = 0; (j < group_size) && (i*group_size+j < local_weight.size()); j++ ){
                    local_weight[i * group_size + j] *= loss_tp;
                }
            }
        }
    }
};

#endif