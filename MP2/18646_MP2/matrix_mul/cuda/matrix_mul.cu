/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#define BLOCK_WIDTH 32
#define BLOCK_X 32
#define BLOCK_Y 32
#define OPTIMIZED 1
#ifdef OPTIMIZED
namespace cuda
{

    static inline int nextPowerOfTwo(int n) {
      n--;

      n = n >>  1 | n;
      n = n >>  2 | n;
      n = n >>  4 | n;
      n = n >>  8 | n;
      n = n >> 16 | n;
      //n = n >> 32 | n;    //  For 64-bit ints

      return ++n;
    }
    __global__ void matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension, int out_dimension){
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int row_sq = row * sq_dimension;
        if(row < sq_dimension && col < sq_dimension) {
          float result = 0.0;
          float matrix_1_data = 0.0;
          float matrix_2_data = 0.0;
          
          extern __shared__ unsigned int sq_1[32 * 32];
          extern __shared__ unsigned int sq_2[32 * 32];
          
          unsigned int * sq_1_idx = (unsigned int *) sq_1;
          unsigned int * sq_2_idx = (unsigned int *) sq_2;
          

          
            
            __syncthreads();
            
            for(int k = 0; k < sq_dimension; k++){	
                //result += sq_1[row * sq_dimension + k] * sq_2[k * sq_dimension + col];
                //sq_matrix_result[row*sq_dimension + col] += sq_1[row * sq_dimension + k] * sq_2[k * sq_dimension + col];
              matrix_1_data = sq_matrix_1[row * sq_dimension + k];
              matrix_2_data = sq_matrix_2[k * sq_dimension + col];
              /*for(int i = 0; i < matrix_2_data; i++) {
                result += matrix_1_data;
              }*/
              result += matrix_1_data * matrix_2_data;
            }
            sq_matrix_result[row*sq_dimension + col] = result;
        }
    }

    void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension){
        int out_size =  (sq_dimension) * (sq_dimension) * sizeof(float);
        int size = (sq_dimension) * (sq_dimension) * sizeof(float);
        float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;

        /***************************************************
        Step 1: Allocation of memory on device memory  
        ****************************************************/

        /* copy sq_matrix_1 and sq_matrix_2 to device memory */
        cudaMalloc((void**) &sq_matrix_1_d, size);
        cudaMemset(sq_matrix_1_d, 0, size);
        cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &sq_matrix_2_d, size);
        cudaMemset(sq_matrix_2_d, 0, size);
        cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);

        /*allocate sq_matrix_result on host */
        cudaMalloc((void**) &sq_matrix_result_d, size);
        cudaMemset(sq_matrix_result_d, 0, size);
        /***************************************************
        Step 2: Invoke kernel 
        ****************************************************/
        int blockNum = ceil(sq_dimension * 1.0 / BLOCK_WIDTH);
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 dimGrid(blockNum, blockNum);

        matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, (sq_dimension), sq_dimension);

        /***************************************************
        Step 3: Transfer result from device to host 
        ****************************************************/
        cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
        cudaFree(sq_matrix_1_d);
        cudaFree(sq_matrix_2_d);
        cudaFree(sq_matrix_result_d);
    }  
} // namespace cuda

#else
namespace cuda
{

    static inline int nextPowerOfTwo(int n) {
      n--;

      n = n >>  1 | n;
      n = n >>  2 | n;
      n = n >>  4 | n;
      n = n >>  8 | n;
      n = n >> 16 | n;
      //n = n >> 32 | n;    //  For 64-bit ints

      return ++n;
    }
    __global__ void matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension, int out_dimension){
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;



        if(row < sq_dimension && col < sq_dimension) {
          for(int k = 0; k < sq_dimension; k++){	
              sq_matrix_result[row*sq_dimension + col] += sq_matrix_1[row * sq_dimension + k] * sq_matrix_2[k * sq_dimension + col];
          }
        }
    }

    void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension){
        int out_size =  (sq_dimension) * (sq_dimension) * sizeof(float);
        int size = (sq_dimension) * (sq_dimension) * sizeof(float);
        float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;

        /***************************************************
        Step 1: Allocation of memory on device memory  
        ****************************************************/

        /* copy sq_matrix_1 and sq_matrix_2 to device memory */
        cudaMalloc((void**) &sq_matrix_1_d, size);
        cudaMemset(sq_matrix_1_d, 0, size);
        cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &sq_matrix_2_d, size);
        cudaMemset(sq_matrix_2_d, 0, size);
        cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);

        /*allocate sq_matrix_result on host */
        cudaMalloc((void**) &sq_matrix_result_d, size);
        cudaMemset(sq_matrix_result_d, 0, size);
        /***************************************************
        Step 2: Invoke kernel 
        ****************************************************/
        int blockNum = ceil(sq_dimension * 1.0 / BLOCK_WIDTH);
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 dimGrid(blockNum, blockNum);

        matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, (sq_dimension), sq_dimension);

        /***************************************************
        Step 3: Transfer result from device to host 
        ****************************************************/
        cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
        cudaFree(sq_matrix_1_d);
        cudaFree(sq_matrix_2_d);
        cudaFree(sq_matrix_result_d);
    }  
} // namespace cuda
#endif