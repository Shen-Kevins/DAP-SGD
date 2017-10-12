#include<iostream>
#include<armadillo>
#include "./Datapoint/Datapoint.h"
#include "./Model/L2L1Model.h"
#include "./Model/L2GroupLassoModel.h"
#include "./Model/L2FusedLassoModel.h"
#include "./Model/L2NuclearLassoModel.h"
#include "./Updater/Updater.h"
#include "./Updater/SGDUpdater.h"
#include "./Run.h"
#include "./Utils/Utils.h"

#include<time.h>


using namespace std;
using namespace arma;

int main(int argc,char **argv){

    srand((unsigned)time(NULL));

    // Datapoint *datapoint = new Datapoint("../data/generated/");

    // Model *model = new L2L1Model();

    // cout<<model->ComputeLoss(datapoint) <<endl;

    // Updater *updater = new SGDUpdater(model, datapoint);

    // clock_t start,end;

    // start = clock();

    // for(int i = 0; i< 10000;i++){
    //     mat grad = updater->ApplySGD();
    //     vector<double> grad_vec = mat2vec(grad);
    //     grad = vec2mat(grad_vec,grad.n_rows,grad.n_cols);

    //     model->UpdateWeight(learning_rate,grad);
    //     mat &local_model = model->GetWeight();
    //     model->ProxyOperator(local_model, L1_lambda * learning_rate);
    //     if(i%1000==0){
    //         cout<<model->ComputeLoss(datapoint) <<endl;
    //         end = clock();
            
    //         cout<<"\t used time:"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    //         start = clock();
    //     }
    // }

    // end = clock();

    // cout<<"used time:"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    
    

    int taskid,numtask;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtask);

    num_workers = numtask - 1;

    RunOnce<L2L1Model,Datapoint>(taskid);

    MPI_Finalize();
    return 0;
}