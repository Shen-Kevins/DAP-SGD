#ifndef _DAPWORKERTRAINER_
#define _DAPWORKERTRAINER_

#include "./Trainer.h"
#include "../define.h"
#include "../Utils/Utils.h"

class DAP_WorkerTrainer: public Trainer{
public:
    DAP_WorkerTrainer(Model *model, Datapoint *datapoint) : Trainer(model,datapoint){}

    void Train() override{
        MPI_Status status;

        vector<double> worker_weight(model->GetSize() * model->GetSize2(),0);

        vector<double> tmp(model->GetSize()* model->GetSize2(),0);


        while(true){

            worker_weight = model->GetWeight();

            

            for(int i=0;i<model->GetSize() * model->GetSize2() ;i++) tmp[i] = worker_weight[i];
            

            vector<double> worker_gradient = updater->ApplySGD();

            model->UpdateWeight(learning_rate,worker_gradient);
            // model->pool->enqueue(model->UpdateWeight,model->weight,learning_rate,worker_gradient);

            model->ProxyOperator(model->GetWeight(), learning_rate * L1_lambda);
            // std::cout<<"a"<<std::endl;
            // std::vector<double> worker_gradient_vec = mat2vec(model->GetWeight() - tmp);
            worker_weight = model->GetWeight();

            for(int i = 0; i < model->GetSize() * model->GetSize2();i++) tmp[i] = worker_weight[i] - tmp[i];
            
            MPI_Send(&tmp[0], tmp.size(), MPI_DOUBLE,0,101,MPI_COMM_WORLD);
            
            
            MPI_Recv(&worker_weight[0],worker_weight.size(),MPI_DOUBLE,0,102,MPI_COMM_WORLD,&status);

            if(worker_weight[0] == 10000)
                break;

            // worker_weight = vec2mat(worker_gradient_vec,worker_weight.n_rows,worker_weight.n_cols);

            model->SetWeight(worker_weight);
        }
    }   
};

#endif