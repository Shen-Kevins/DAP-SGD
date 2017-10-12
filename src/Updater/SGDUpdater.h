#ifndef _SGDUPDATER_
#define _SGDUPDATER_

#include "./Updater.h"
#include<cstdlib>

class SGDUpdater : public Updater{
public:
    SGDUpdater() : Updater(){}

    SGDUpdater(Model *model, Datapoint *datapoint) : Updater(model, datapoint){}

    vector<double> ApplySGD() override{
        int begin = rand()%(datapoint->GetSize()-mini_batch);
        mat gradient;
        // if(datapoint->GetFeature().n_cols == 784){
            // gradient = logistic_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),model->GetWeight(),datapoint->GetLabelsRols(begin,begin+mini_batch));
        // }
        // else{
            gradient = least_backward(datapoint->GetFeaturesRows(begin,begin+mini_batch),vec2mat(model->GetWeight(),model->GetSize(),model->GetSize2()),datapoint->GetLabelsRols(begin,begin+mini_batch));
        // }
        vector<double> gradients = mat2vec(gradient);
        // gradient += L2_lambda * model->GetWeight();

        for(int i = 0; i < gradients.size(); i++){
            gradients[i] += L2_lambda * model->GetWeight()[i];
        }

        // model->UpdateWeight(learning_rate,gradient);
        return gradients;
    }
};

#endif
