/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

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

#include <omp.h>
#include "matrix_mul.h"
#include <string.h>
#define IMPROVED

namespace omp{
    void  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension ){

    #ifndef IMPROVED
        #pragma omp parallel for 
        for (unsigned int i = 0; i < sq_dimension; i++){
        
            for(unsigned int j = 0; j < sq_dimension; j++){       
                sq_matrix_result[i*sq_dimension + j] = 0;
                    for (unsigned int k = 0; k < sq_dimension; k++) {
                    
                    sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
                }
            }
        }// End of parallel region
    #else
        memset(sq_matrix_result,0,sq_dimension*sq_dimension*sizeof(float));
        #pragma omp parallel 
            { 
                int id = omp_get_thread_num();
                int threads = omp_get_num_threads();
                for (unsigned int k = 0; k < sq_dimension; k++) {
                    int offset = k*sq_dimension;
                    for (unsigned int i = id; i < sq_dimension; i+=threads){
                        int ind = i*sq_dimension;
                        #pragma unroll(4)
                        for(unsigned int j = 0; j < sq_dimension; j++){       
                            
                            
                            sq_matrix_result[ind + j] += sq_matrix_1[ind + k] * sq_matrix_2[offset + j];
                        }
                    }
                }
            }// End of parallel region
    #endif
  }
  
} //namespace omp
