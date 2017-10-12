#ifndef _DEFINE_
#define _DEFINE_

#include<mpi.h>

double L1_lambda = 0.001;
double GroupLasso_lambda = 0.001;
double FusedLasso_lambda = 0.001;
double NuclearLasso_lambda = 0.001;
double L2_lambda = 0.00001;
double learning_rate = 0.0003;//除了grouplasso（0.0003）  都是0.0007 
double learning_rate_decay = 0;

int mini_batch = 1;


int epoch_num = 10;
int in_inter = 12665;//1000

int num_workers = 2;

int max_delay = 20;


// #include "./Datapoint/Datapoint.h"
// #include "./Updater/SGDUpdater.h"
// #include "./Model/L2FusedLassoModel.h"
// #include "./Model/L2GroupLassoModel.h"
// #include "./Model/L2L1Model.h"
// #include "./Utils/Utils.h"
// #include "./Trainer/ServerTrainer.h"
// #include "./Trainer/WorkerTrainer.h"

#endif