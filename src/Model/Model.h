#ifndef _MODEL_
#define _MODEL_

#include<iostream>
#include<armadillo>
#include "../Datapoint/Datapoint.h"
#include "../define.h"
#include "../Utils/Utils.h"
#include "../Utils/ThreadPool.h"

using namespace std;
using namespace arma;

class Model{
protected:
    // mat weight;

    
    int size;
    int size2;

    

public:
    vector<double> weight;
    ThreadPool *pool;

    Model(int length){
        // arma_rng::set_seed(1);
        // this->weight.randn(length,1);
        // this->weight = weight / 10.0;

        weight.resize(length,0.001);
        size = length;
        size2 = 1;

        // model.print();
        pool = new ThreadPool(2);
    }

    int GetSize(){
        return this->size;
    }

    int GetSize2(){
        return this->size2;
    }

    vector<double>& GetWeight(){
        return this->weight;
    }

    void SetWeight(vector<double> &weights){
        this->weight = weights;
    }

    double ComputeL2Loss(){
        double L2loss = 0.0;
        for(int i = 0; i < this->size; i++){
            L2loss += weight[i] * weight[i];
        }
        // return 0.5 * L2_lambda * accu(square(this->weight));

        return 0.5 * L2_lambda * L2loss;
    }

    // void Update(double learning_rate,vector<double> gradient){
    //     this->pool->enqueue(UpdateWeight,weight,learning_rate,gradient);
    // }

    void UpdateWeight(double learning_rate,vector<double> &gradient){

        // int split = weight.size() / 2;

        // vector<double> &weight = this->weight;

        // // mutex g_lock;

        // for (int i=0;i<2;++i)
        // {
        //     pool->enqueue(
        //                  [&weight, &gradient, &learning_rate](int l, int r)mutable
        //                  {
        //                      for (int j=l;j<r;++j) 
        //                     {   
        //                         // g_lock.lock();
        //                         weight[j] -= learning_rate*gradient[j];
        //                         // g_lock.unlock();
        //                     }
        //                 }
        //                  , i*split, (i+1)*split
        //                  );
        // }

        for(int i = 0; i < weight.size(); i++)
            weight[i] -=learning_rate * gradient[i];
    }

    // void Save_weight(){
    //     this->weight.save("../../data/generated/least_train_weight_mat");
    // }

    virtual double ComputeRegularization() = 0;

    virtual double ComputeLoss(Datapoint *datapoint) = 0;

    virtual void ProxyOperator(vector<double> &local_weight, double gama) = 0;

};

#endif