#ifndef _TRAINER_
#define _TRAINER_

#include "../Model/Model.h"
#include "../Datapoint/Datapoint.h"
#include "../Updater/Updater.h"
#include<time.h>

class Trainer{
protected:
    Model *model;
    Datapoint *datapoint;
    Updater *updater;

public:
    Trainer(){}

    Trainer(Model *model, Datapoint *datapoint){
        this->model = model;
        this->datapoint = datapoint;
        this->updater = new SGDUpdater(model,datapoint);
    }

    virtual void Train() = 0;


};


#endif