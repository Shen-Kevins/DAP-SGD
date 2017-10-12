#ifndef _L2FUSEDLASSOMODEL_
#define _L2FUSEDLASSOMODEL_

#include "./Model.h"

class L2FusedLassoModel : public Model{
public:
    L2FusedLassoModel(int length) : Model(length){}

    double ComputeRegularization() override{
        double regloss = 0.0;
        for(int i = 0; i < this->size - 1; i++){
            regloss += std::abs(this->weight[i+1] - this->weight[i]);
        }
        return regloss * FusedLasso_lambda;
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
        double inner_eta = 1.0/(2.0 - 2.0 * cos((local_weight.size()-1))*M_PI/local_weight.size());
        double* z_ = new double[local_weight.size()];
        double* z_temp_ = new double[local_weight.size()];
        memset(z_,0，(local_weight.size()-1)*sizeof(double));
    
        int max_iter_ = 10;//这里的最大迭代次数是自己定的
        double temp;
    
        for(int t = 0; t < max_iter_;t++){
            memcpy(z_temp_,z_,(local_weight.size()-1)*sizeof(double));
    
            for(int i = 0; i < local_weight.size()-1;i++){
                temp = 2*z_temp_[i];
                if(0 < i) temp -= z_temp_[i-1];
                if(i+1 < local_weight.size()-1) temp -= z_temp_[i+1];
                z_[i] -= inner_eta * (temp - (local_weight[i]-local_weight[i+1]));
            }
    
            for(int i = 0; i < local_weight.size()-1;i++){
                if(z_[i] >= 0)
                    // z_[i] = min(z_[i], gamma);
                    z_[i] = z_[i] < gama? z_[i] : gama;
                else
                    // z_[i] = max(z_[i],-gamma);
                    z_[i] = z_[i] > gama? z_[i] : gama;
            }
        }
    
        for(int i = 0;i < local_weight.size();i++){
            temp = 0;
            if(0 < i) temp -= z_[i-1];
            if(i < local_weight.size()-1) temp += z_[i];
            local_weight[i] -= temp;
        }
    }
};

#endif