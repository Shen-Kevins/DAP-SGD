#include <iostream>
#include <vector>
#include <chrono>

#include<time.h>

#include <sys/time.h>

#include "./Utils/ThreadPool.h"
#include<pthread.h>

using namespace std;

int main()
{
    int n = 2;
    
    pthread_t threads[2];
    
    std::vector<double> weight(5000,1);
    std::vector<double> grad(5000,1);
    double lambda = 0.5;

    size_t split = weight.size() / n;
    // size_t = 
    auto start = std::chrono::high_resolution_clock::now();
    for(int k = 0; k < 1000;k++)
    for(int i = 0; i < weight.size(); i++){
        weight[i] += lambda * grad[i];
    }

    auto mid = std::chrono::high_resolution_clock::now();
    // for(int k = 0; k < 100;k++)
    time_t start1,end1;
    // start1 = time(NULL);
    // start1 = clock();
    ThreadPool pool(2);

    struct timeval t_start,t_end;
    
        long cost_time = 0;
    
     
    
    //get start time
    
    gettimeofday(&t_start, NULL);
    
    printf("Start time: %ld us\n", t_start.tv_usec);
for(int k=0;k<10000;k++)
    for (int i=0;i<2;++i)
    {
        
        pool.enqueue(
                     [&weight, &grad, &lambda](int l, int r)mutable
                     {
                         for (int j=l;j<r;++j) 
                        {   
                            weight[j] += lambda*grad[j];
                        }
                        // sleep(1);
                        // std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                     , i*split, (i+1)*split
                     );
    }
    pool.wait();
    // end1 = time(NULL);
    // end1 = clock();

    gettimeofday(&t_end, NULL);
    
    printf("End time: %ld us\n", t_end.tv_usec);
    
     
    
    //calculate time slot
    
    cost_time = t_end.tv_usec - t_start.tv_usec;
    
    printf("Cost time: %ld us\n", cost_time);

    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff1 = mid-start;
    std::chrono::duration<double> diff2 = end-mid;
    std::cout<<diff1.count()<<" "<<diff2.count()<<"\n";
    // cout << end1-start1<<endl;
}