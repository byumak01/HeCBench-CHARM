/**********************************************************************
 * Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ********************************************************************/

#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

int main(int argc, char *argv[])
{
  if(argc != 2)
    {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }

  const int iterations
    = atoi(argv[1]); // Number of iterations for kernel execution
  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool fail = false;

  if(iterations < 1)
    {
      std::cout << "Iterations cannot be 0 or negative. Exiting..\n";
      return -1;
    }

  if(treeNumber < GROUP_SIZE)
    {
      std::cout << "treeNumber should be larger than the work group size"
                << std::endl;
      return -1;
    }
  if(treeNumber % 256 != 0)
    {
      std::cout << "treeNumber should be a multiple of 256" << std::endl;
      return -1;
    }

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  // input data
  int *data = (int *)malloc(inputSize);

  // device result
  int *deviceResult = (int *)malloc(outputSize);

  // compute reference for verification
  int *reference = (int *)malloc(outputSize);
  memset(reference, 0, outputSize);
  for(int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

  sycl::queue q;

  sycl::range<1> gws(treeNumber);
  sycl::range<1> lws(GROUP_SIZE);

  sycl::nd_range<3> ndr(sycl::range<3>(1, 1, treeNumber), // Global range
                        sycl::range<3>(1, 1, GROUP_SIZE)  // Local range
  );

  sycl::buffer<int, 1> inputBuffer(data, elements);
  sycl::buffer<int, 1> outputBuffer(deviceResult, treeNumber);

  // initialize aos data
  for(int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[j + i * treeSize] = j + i * treeSize;

  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
    {
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read> AppleTree(inputBuffer,
                                                                  cgh);
        sycl::accessor<int, 1, sycl::access_mode::write> Result(outputBuffer,
                                                                cgh);

        cgh.parallel_for(ndr, [=](sycl::nd_item<3> const &item) {
          int gid = item.get_global_linear_id(); // Use the Z-dimension
                                                 // as the 1D index
          int res = 0;
          for(int j = 0; j < treeSize; j++)
            {
              res += AppleTree[gid * treeSize + j];
            }
          Result[gid] = res;
        });
      });
    }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time
    = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (AoS): "
            << (time * 1e-3f) / iterations << " (us)\n";

   sycl::host_accessor<int, 1, sycl::access_mode::read> hostResult(
     outputBuffer);

  for(int i = 0; i < treeNumber; i++)
    {
      if(deviceResult[i] != reference[i])
        {
          fail = true;
          break;
        }
    }

  if(fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  // initialize soa data
  for(int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[i + j * treeNumber] = j + i * treeSize;

  start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
    {
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read> AppleTree(inputBuffer,
                                                                  cgh);
        sycl::accessor<int, 1, sycl::access_mode::write> Result(outputBuffer,
                                                                cgh);
        cgh.parallel_for(ndr, [=](sycl::nd_item<3> const &item) {
          uint gid = item.get_global_linear_id();
          uint res = 0;
          for(int j = 0; j < treeSize; j++)
            {
              res += AppleTree[j * gid + treeSize];
            }
          Result[gid] = res;
        });
      });
    }

  q.wait();
  end = std::chrono::steady_clock::now();
  time
    = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (SoA): "
            << (time * 1e-3f) / iterations << " (us)\n";

  //sycl::host_accessor<int, 1, sycl::access_mode::read> hostResult(
  //  outputBuffer);

  for(int i = 0; i < treeNumber; i++)
    {
      if(deviceResult[i] != reference[i])
        {
          fail = true;
          break;
        }
    }

  if(fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";
  /*
    sycl::free(inputBuffer, q);
    sycl::free(outputBuffer, q);
    free(deviceResult);
    free(reference);
    free(data);
  */
  return 0;
}
