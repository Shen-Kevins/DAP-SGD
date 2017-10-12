#ifndef _RUN_
#define _RUN_

#include<iostream>
#include "./define.h"
#include "./Model/Model.h"
#include "./Datapoint/Datapoint.h"
#include "./Datapoint/NuclearDatapoint.h"
#include "./Datapoint/LogisticDatapoint.h"
#include "./Updater/Updater.h"
#include "./Updater/SGDUpdater.h"
#include "./Trainer/TAP_ServerTrainer.h"
#include "./Trainer/TAP_WorkerTrainer.h"
#include "./Trainer/DAP_ServerTrainer.h"
#include "./Trainer/DAP_WorkerTrainer.h"

template<class MODEL_CLASS, class DATAPOINT_CLASS>
void RunOnce(int taskid){
    Model *model = new MODEL_CLASS(5000);
    Datapoint *datapoint = new DATAPOINT_CLASS("../data/generated/");

    Updater *updater = new SGDUpdater(model,datapoint);

    Trainer *trainer = NULL;
    if(taskid == 0){
        trainer = new DAP_ServerTrainer(model,datapoint);
        trainer->Train();
    }else{
        trainer = new DAP_WorkerTrainer(model,datapoint);
        trainer->Train();
    }

    // trainer->Train();
}


#endif