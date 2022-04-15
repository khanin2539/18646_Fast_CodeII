#include "linear.h"
#include "../utils/utils.h"
#include "def.h"


__global__
void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;
    // N_IN N_HIDDEN followed from main as input layers and hidden layers
    /*
    extern  __shared__ float shared_weights[N_IN][N_HIDDEN];
    extern  __shared__ float shared_bias[N_HIDDEN];
    // shared weights are good

    if ((row < bs) && (col < n_out)){
        //update shared mem 
        for(int i = threadIdx.x; i <n_in*n_out; i += blockDim.x){
            
            shared_weights[i] = weights[i];
            
        }
            
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) for (int i = 0; i < n_out; i++) shared_bias[i] = bias[i];
    __syncthreads();
    if ...{
        ind_out = row*n_out + col;
//         out[ind_out] = bias[col];
        float local_sum = shared_bias[col];
        ind_inp = row*n_in;
        ind_weights = col;
        for (int i=0; i<n_in; i++){
            
            
            local_sum += inp[ind_inp]*shared_weights[ind_weights];
            ind_inp +=1;
            ind_weights += n_out;
        }
        out[ind_out] += local_sum;
        
    }
    */
    float out_ind;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        out_ind = bias[col];
        ind_inp = row*n_in;
        ind_weights = col;
        for (int i=0; i<n_in; i++){
            
            out_ind += inp[ind_inp]*weights[ind_weights];
            ind_inp +=1;
            ind_weights += n_out;
        }
        out[ind_out] = out_ind;
    }

    
    
}


__global__
void linear_backward_gpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        ind_inp = row*n_in;
        ind_weights = col;
        for (int i=0; i<n_in; i++){
            

            atomicAdd(&inp[ind_inp], weights[ind_weights]*out[ind_out]);
            ind_inp +=1;
            ind_weights += n_out;
        }
    }
}


__global__
void linear_update_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr){
    int row = blockDim.x*blockIdx.x + threadIdx.x, col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)){
        ind_out = row*n_out + col;
        atomicAdd(&bias[col], -lr*out[ind_out]);
        ind_inp = row*n_in;
        ind_weights = col;
        for (int i=0; i<n_in; i++){
            

            atomicAdd(&weights[ind_weights], -lr*inp[ind_inp]*out[ind_out]);
            ind_inp +=1;
            ind_weights += n_out;
        }
    }
}


Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr){
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in*n_out;
    sz_out = bs*n_out;
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size - 1) / block_size;

    cudaMallocManaged(&weights, sz_weights*sizeof(float));
    cudaMallocManaged(&bias, n_out*sizeof(float));

    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}


void Linear_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
}


void Linear_GPU::backward(){
    init_zero(inp, bs*n_in);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
    cudaDeviceSynchronize();

    cudaFree(cp_weights);
}


void Linear_GPU::update(){
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    set_eq(cp_weights, weights, sz_weights);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);
    
    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaDeviceSynchronize();
}
void Linear_GPU::free(){
    cudaFree(weights);
    cudaFree(bias);
    cudaFree(cp_weights);
    
}
