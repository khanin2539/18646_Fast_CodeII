#ifndef TRAIN_GPU_H
#define TRAIN_GPU_H


#include "../OLD_GPU/sequential.h"


void train_oldgpu(Sequential_OLDGPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs);


#endif
