#ifndef _L2NUCLEARLASSOMODEL_
#define _L2NUCLEARLASSOMODEL_

#include "./Model.h"

class L2NuclearLassoModel : public Model{
public:
    L2NuclearLassoModel(int length) : Model(length){
        // arma_rng::set_seed(1);
        // this->weight.randn(50,40);
        // this->weight = this->weight / 10.0;

        this->weight.resize(50*40,0.001);
        this->size = 50;
        this->size2 = 40;
    }

    double ComputeRegularization() override{
        double regloss = 0.0;
        mat U;
        vec s;
        mat V;
        svd(U,s,V,vec2mat(this->weight,50,40));
        for(int i = 0; i < s.size(); i++){
            regloss += std::abs(s(i));
        }
        return regloss * NuclearLasso_lambda;
    }

    double ComputeLoss(Datapoint *datapoint) override{
        double loss = 0;
        if(this->size == 784){
            loss = logistic_forward(datapoint->GetFeature(),vec2mat(this->weight,50,40),datapoint->GetLabel());
        }
        else{
            // loss = accu(square(datapoint->GetFeature() * weight-(datapoint->GetLabel()))) /(2 * datapoint->GetSize());
            loss = least_forward(datapoint->GetFeature() , vec2mat(weight,50,40),datapoint->GetLabel());
        }
        return loss + ComputeL2Loss() + ComputeRegularization();
    }

    void ProxyOperator(vector<double> &local_weight, double gama) override{
        mat U;
        vec s;
        mat V;
        svd(U,s,V,vec2mat(local_weight,50,40));
        for(int i = 0; i < s.size(); i++){
            double val = s(i);
            double sign = val > 0? 1:-1;
            s(i) = sign * fmax(std::abs(val)-gama, 0);
        }

        mat S;
        S.zeros(50, 40);
        for(int i = 0;i < s.size();i++){
            S(i, i) = s(i);
        }
        local_weight = mat2vec(U * S * V.t());
    }

};

#endif
