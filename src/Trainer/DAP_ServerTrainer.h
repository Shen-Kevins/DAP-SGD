#ifndef _DAPSERVERTRAINER_
#define _DAPSERVERTRAINER_

#include "../define.h"
#include "./Trainer.h"
#include "../Utils/Utils.h"
#include<fstream>
#include<iostream>
#include<list>

using namespace std;

class DAP_ServerTrainer : public Trainer{
public:

    DAP_ServerTrainer(Model *model, Datapoint *datapoint) : Trainer(model,datapoint){}

    void Train() override{
        double sum_time = 0.0;
        clock_t start,end;

        list<vector<double> > weight_data_list;
        list<double> time_list;
        weight_data_list.push_front(model->GetWeight());
        time_list.push_front(0);

        MPI_Status status;

        printf("Epoch:  Times(s)\n");

        // mat worker_gradient(model->GetWeight().n_rows,model->GetWeight().n_cols,fill::zeros);
        // std::vector<double> worker_gradient_vec(worker_gradient.size(),0);

        std::vector<double> worker_gradient(model->GetSize() * model->GetSize2(),0);
        std::vector<double> master_weight(model->GetSize() * model->GetSize2(),0);

        for(int epoch = 0; epoch < epoch_num; epoch++){
            // if(epoch == 50){
            //     learning_rate /= 10;
            // }
            start = clock();
            double learning_rates = learning_rate / std::pow(1+epoch, learning_rate_decay);

            std::vector<int> delay_counter(num_workers, 1);


            for(int inter_counter = 0 ; inter_counter < in_inter; ){
                std::vector<int> cur_received_workers(num_workers, 0);

                bool flag_receive = true;

                while(flag_receive){
                    MPI_Probe(MPI_ANY_SOURCE,101,MPI_COMM_WORLD,&status);
                    int taskid = status.MPI_SOURCE;

                    if(delay_counter[taskid - 1] > max_delay){
                        delay_counter[taskid - 1] = 1;
                        cur_received_workers[taskid - 1] = 0;
                        flag_receive = true;
                        std::cout << "delay\n"<<std::endl;
                        MPI_Recv(&worker_gradient[0],worker_gradient.size(),MPI_DOUBLE,taskid,101,MPI_COMM_WORLD,&status);
                        
                        master_weight = model->GetWeight();
                        MPI_Send(&master_weight[0], master_weight.size(),MPI_DOUBLE,taskid,102,MPI_COMM_WORLD);

                        // inter_counter--;
                        continue;
                        
                    }
                    
                    MPI_Recv(&worker_gradient[0],worker_gradient.size(),MPI_DOUBLE,taskid,101,MPI_COMM_WORLD,&status);
                    // worker_gradient = vec2mat(worker_gradient_vec,worker_gradient.n_rows,worker_gradient.n_cols);
                    
                    model->UpdateWeight(-1, worker_gradient);
                    // model->pool->enqueue(model->UpdateWeight,model->weight,-1,worker_gradient);
                    //proxy操作，这里先写在master上，暂时默认L1
                    // model->ProxyOperator(model->GetWeight(), learning_rate * L1_lambda);

                    delay_counter[taskid - 1] = 1;
                    cur_received_workers[taskid - 1] = 1;
                   
                    flag_receive = false;

                }

                master_weight = model->GetWeight();
                // worker_gradient_vec = mat2vec(master_weight);

                for(int i = 0; i < num_workers; i++){
                    if(cur_received_workers[i] == 0)
                        delay_counter[i] += 1;
                    else{

                        MPI_Send(&master_weight[0], master_weight.size(),MPI_DOUBLE,i+1,102,MPI_COMM_WORLD);
                    }
                }
                inter_counter++;
            }

            end = clock();
            sum_time += double(end-start) / CLOCKS_PER_SEC;

            printf("%d\t%f\n",epoch,sum_time);

            // model->Save_weight();

            weight_data_list.push_front(master_weight);
            time_list.push_front(sum_time);

        }

        for(int i = 0; i < num_workers; i++){
            MPI_Probe(MPI_ANY_SOURCE,101,MPI_COMM_WORLD,&status);
            int taskid = status.MPI_SOURCE;
            MPI_Recv(&worker_gradient[0],worker_gradient.size(),MPI_DOUBLE,taskid,101,MPI_COMM_WORLD,&status);
            master_weight[0] = 10000;
            MPI_Send(&master_weight[0], master_weight.size(),MPI_DOUBLE,taskid,102,MPI_COMM_WORLD);

        }


        printf("please waiting, loss is calculating.....\n");
        printf("Epoch   tims(s)       loss\n");

        fstream outfile("../data/result/nuclear_lasso_dap_10.txt", ios::out);

        int size = weight_data_list.size();
        for(int i = 0; i < size; i++){
            // master_weight = weight_data_list.back();
            model->SetWeight(weight_data_list.back());
            weight_data_list.pop_back();
            
            double loss = model->ComputeLoss(datapoint);
            double time = time_list.back();

            time_list.pop_back();

            printf("%d\t%f\t%f\n",i,time,loss);
            outfile << i<<"\t"<<time<<"\t"<<loss<<"\n";
        }
        outfile.close();  

    }
};

#endif