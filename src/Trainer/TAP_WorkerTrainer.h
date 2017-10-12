#ifndef _TAPWORKERTRAINER_
#define _TAPWORKERTRAINER_

#include "./Trainer.h"
#include "../define.h"
#include "../Utils/Utils.h"

class TAP_WorkerTrainer: public Trainer{
public:
    TAP_WorkerTrainer(Model *model, Datapoint *datapoint) : Trainer(model,datapoint){}

    void Train() override{
        MPI_Status status;

        vector<double> worker_weight(model->GetSize() * model->GetSize2(),0);
        while(true){
            vector<double> worker_gradient = updater->ApplySGD();
            // std::cout<<"a"<<std::endl;
            // std::vector<double> worker_gradient_vec = mat2vec(worker_gradient);
            MPI_Send(&worker_gradient[0], worker_gradient.size(), MPI_DOUBLE,0,101,MPI_COMM_WORLD);
            
            
            MPI_Recv(&worker_weight[0],worker_weight.size(),MPI_DOUBLE,0,102,MPI_COMM_WORLD,&status);

            if(worker_weight[0] == 10000)
                break;

            // worker_weight = vec2mat(worker_gradient_vec,worker_weight.n_rows,worker_weight.n_cols);

            model->SetWeight(worker_weight);
        }
    }   
};

#endif