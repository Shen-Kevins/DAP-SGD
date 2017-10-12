#include<iostream>
#include<sys/time.h>
#include <vector>
#include<cmath>
using namespace std;

int main(){
    std::vector<double> weight(5000,1);
    std::vector<double> grad(5000,1);
    double lambda = 0.5;

    struct timeval t_start,t_end,t_mid;
    
        long cost_time = 0;
    
     
    
    //get start time
    
    gettimeofday(&t_start, NULL);
    
    // printf("Start time: %d us\n", t_start.tv_usec);
    

    for(int k = 0; k < 1000;k++)
    for(int i = 0; i < weight.size(); i++){
        weight[i] += lambda * grad[i];
    }
    
    gettimeofday(&t_mid, NULL);
    //calculate time slot
    // printf("Mid time: %d us\n", t_mid.tv_usec);

    
    
    // printf("Cost time1: %ld us\n", cost_time);


    for(int k = 0; k < 1000;k++){
    for(int i = 0; i < weight.size(); i++){
        weight[i] = grad[i];
    // }
    // int group_size = 10;
    // int group_num = weight.size() / group_size +(weight.size() % group_size != 0);
    // for(int i = 0; i < group_num; i++){
    //     double loss_tp = 0;
    //     for(int j = 0; (j < group_size) && (i*group_size+j < weight.size()); j++ ){
    //         loss_tp += weight[i * group_size + j] * weight[i * group_size + j];
    //     }
    //     loss_tp = std::sqrt(loss_tp);

    //     if(loss_tp <= lambda){
    //         for(int j = 0; (j < group_size) && (i*group_size+j < weight.size()); j++ ){
    //             weight[i * group_size + j] = 0.0;
    //         }
    //     }else{
    //         loss_tp = 1 - lambda/loss_tp;
    //         for(int j = 0; (j < group_size) && (i*group_size+j < weight.size()); j++ ){
    //             weight[i * group_size + j] *= loss_tp;
    //         }
    //     }
    }
    }

    gettimeofday(&t_end, NULL);
    //calculate time slot
    // printf("End time: %d us\n", t_end.tv_usec);

    cost_time = t_mid.tv_usec - t_start.tv_usec;
    printf("Cost time1: %ld us\n", cost_time);

    cost_time = t_end.tv_usec - t_mid.tv_usec; 
    printf("Cost time2: %ld us\n", cost_time);


    return 0;
}