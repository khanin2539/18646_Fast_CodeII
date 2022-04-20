# vim:set ts=8 sw=8 sts=0 noet:

#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for programs running a simple k-means clustering */
#                  algorithm                                                 */
#                                                                            */
#    Author:  Wei-keng Liao                                                  */
#             ECE Department Northwestern University                         */
#             email: wkliao@ece.northwestern.edu                             */
#    Copyright, 2005, Wei-keng Liao                                          */
#                                                                            */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

# Copyright (c) 2005 Wei-keng Liao
# Copyright (c) 2011 Serban Giuroiu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ------------------------------------------------------------------------------

.KEEP_STATE:

all: neural

DFLAGS      =
OPTFLAGS    = -g -pg -O2 -std=c++11
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
NVCCFLAGS   = $(CFLAGS) 
LDFLAGS     = $(OPTFLAGS)
LIBS        =

CC          = gcc
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

# ------------------------------------------------------------------------------
# CUDA Version

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

CUDA_C_SRC =
CUDA_CU_SRC = main.cpp ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

CUDA_C_OBJ = $(CUDA_C_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)


LINEAR_C_SRC =   ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

LINEAR_CU_SRC =  ../test/linear_test.cu ../GPU/linear.cu 

LINEAR_C_OBJ = $(LINEAR_C_SRC:%.cu=%.o)
LINEAR_CU_OBJ = $(LINEAR_CU_SRC:%.cu=%.o)


MSE_C_SRC =   ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

MSE_CU_SRC =  ../test/mse_test.cu ../GPU/mse.cu 

MSE_C_OBJ = $(MSE_C_SRC:%.cu=%.o)
MSE_CU_OBJ = $(MSE_CU_SRC:%.cu=%.o)


RELU_C_SRC =   ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

RELU_CU_SRC =  ../test/relu_test.cu ../GPU/relu.cu 

RELU_C_OBJ = $(RELU_C_SRC:%.cu=%.o)
RELU_CU_OBJ = $(RELU_CU_SRC:%.cu=%.o)


SEQ_C_SRC =   ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

SEQ_CU_SRC =  ../test/sequential_test.cu ../GPU/sequential.cu  ../OLD_GPU/linear.cu  ../OLD_GPU/relu.cu 

SEQ_C_OBJ = $(SEQ_C_SRC:%.cu=%.o)
SEQ_CU_OBJ = $(SEQ_CU_SRC:%.cu=%.o)


TRAIN_C_SRC =   ../data/read_csv.cpp ../utils/utils.cpp mse_cpu.cpp relu_cpu.cpp linear_cpu.cpp sequential_cpu.cpp train_cpu.cpp

TRAIN_CU_SRC =  ../test/sequential_test.cu ../GPU/train.cu  ../OLD_GPU/mse.cu  ../OLD_GPU/sequential.cu  ../OLD_GPU/linear.cu  ../OLD_GPU/relu.cu

TRAIN_C_OBJ = $(TRAIN_C_SRC:%.cu=%.o)
TRAIN_CU_OBJ = $(TRAIN_CU_SRC:%.cu=%.o)

neural: neural_main
neural_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(CUDA_C_OBJ) $(CUDA_CU_OBJ)

linear: test_linear
test_linear: $(LINEAR_C_OBJ) $(LINEAR_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(LINEAR_C_OBJ) $(LINEAR_CU_OBJ)

mse: test_mse
test_mse: $(MSE_C_OBJ) $(MSE_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(MSE_C_OBJ) $(MSE_CU_OBJ)

relu: test_relu
test_relu: $(RELU_C_OBJ) $(RELU_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(RELU_C_OBJ) $(RELU_CU_OBJ)

seq: test_seq
test_seq: $(SEQ_C_OBJ) $(SEQ_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(SEQ_C_OBJ) $(SEQ_CU_OBJ)

train: test_train
test_train: $(TRAIN_C_OBJ) $(TRAIN_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(TRAIN_C_OBJ) $(TRAIN_CU_OBJ)

#---------------------------------------------------------------------
clean:
	rm -rf *.o ../GPU/*.o ../OLD_GPU/*.o ../test/*.o neural_main test_linear train_test relu_test sequential_test test_mse test_relu test_seq test_train gmon.out\