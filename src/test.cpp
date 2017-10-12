#include "./Model/L2NuclearLassoModel.h"
#include "./Model/L2L1Model.h"
#include "./Datapoint/Datapoint.h"
#include "./Datapoint/NuclearDatapoint.h"
#include "./Datapoint/LogisticDatapoint.h"
#include "./Updater/SGDUpdater.h"
#include "./Utils/Utils.h"
#include<iostream>
#include<time.h>

using namespace std;

int main(){
  /*  Model *model = new L2NuclearLassoModel();

    double regloss = model->ComputeRegularization();
    cout<<"regloss:"<<regloss<<endl;

    Datapoint *datapoint = new NuclearDatapoint("../data/generated/");
    double loss = model->ComputeLoss(datapoint);
    cout<<"loss:"<<loss<<endl;

    Updater *updater = new SGDUpdater(model, datapoint);

    clock_t start,end;
    start = clock();

    for(int i = 0; i< 10000;i++){
        mat grad = updater->ApplySGD();
        model->UpdateWeight(learning_rate,grad);
        mat &local_model = model->GetWeight();
        model->ProxyOperator(local_model, L1_lambda * learning_rate);
        if(i%1000==0){
            cout<<model->ComputeLoss(datapoint) <<endl;
            end = clock();
            
            cout<<"\t used time:"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
            start = clock();
        }
    }
    */
    Datapoint *datapoint = new LogisticDatapoint("../data/mnist/");
    cout << datapoint->GetSize() << endl;
    Model *model = new L2L1Model(784);

    cout << logistic_forward(datapoint->GetFeature(),model->GetWeight(),datapoint->GetLabel()) << endl;

    mat grad = logistic_backward(datapoint->GetFeature(),model->GetWeight(),datapoint->GetLabel());
    model->UpdateWeight(0.01,grad);

    cout << logistic_forward(datapoint->GetFeature(),model->GetWeight(),datapoint->GetLabel()) << endl;

    return 0;
}