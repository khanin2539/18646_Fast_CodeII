#ifndef TRAIN_OLDOLDGPU_H
#define TRAIN_OLDOLDGPU_H


#include "sequential.h"


void train_oldgpu(Sequential_OLDGPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs);


#endif
