#include <iostream>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "mse.h"
#include "train.h"
#include "../utils/utils.h"


int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

void train_cpu(Sequential_CPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs){
    MSE_CPU mse(bs);
    
    int sz_inp = bs*n_in;
    float *cp_inp = new float[sz_inp], *out;

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);
        
        mse.backward();
        seq.update();

        /*if(i%3 == 0) {
            std::cout << "Physcial RAM used " << getValue() << " KB" << std::endl;
        }*/
    }
    delete[] cp_inp;
    
    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);
    std::cout << "The final loss is: " << targ[bs] << std::endl;
}
