#include <chrono>
#include <iostream>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "kernels_with_range.h"
#include <vector>

// thread block size
#define BLOCK_SIZE 256

template <class T>
void test (sycl::queue &q, const int repeat, const int numFloats)
{
  // Initialize host data, with the first half the same as the second
  //T* hostMem = (T*) malloc (sizeof(T) * numFloats);
  //T* _deviceMem = (T*) malloc (sizeof(T) * numFloats);
  std::vector<T> hostMem(numFloats);
  std::vector<T> _deviceMem(numFloats);
  
  srand48(123);
  for (int j = 0; j < numFloats/2 ; ++j)
    hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);
  
  //sycl::buffer<T, 1> hostMem_buffer(hostMem, numFloats);
  //sycl::buffer<T, 1> deviceMem_buffer(_deviceMem, numFloats);

  sycl::buffer<T, 1> hostMem_buffer(hostMem.data(), sycl::range<1>(numFloats));
  sycl::buffer<T, 1> deviceMem_buffer(_deviceMem.data(), sycl::range<1>(numFloats));

  sycl::range<1> gws (numFloats);
  sycl::range<1> lws (BLOCK_SIZE);


  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Add1<T>(item, deviceMem, repeat, 10.0);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Add2<T>(item, deviceMem, repeat, 10.0);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Add4<T>(item, deviceMem, repeat, 10.0);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Add8<T>(item, deviceMem, repeat, 10.0);
      });
    });
    q.wait();
  }

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }

  auto k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class add1<T>>(gws, [=](sycl::id<1> id) {
      Add1<T>(item, deviceMem, repeat, 10.0);
    });
  });
  q.wait();
  auto k_end = std::chrono::high_resolution_clock::now();
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add1): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }

  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class add2<T>>(gws, [=](sycl::id<1> id) {
      Add2<T>(item, deviceMem, repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add2): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class add4<T>>(gws, [=](sycl::id<1> id) {
      Add4<T>(item, deviceMem, repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add4): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class add8<T>>(gws, [=](sycl::id<1> id) {
      Add8<T>(item, deviceMem, repeat, 10.0);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Mul1<T>(item, deviceMem, repeat, 1.01);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Mul2<T>(item, deviceMem, repeat, 1.01);
      });
    });
    q.submit([&](sycl::handler &cgh) {sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Mul4<T>(item, deviceMem, repeat, 1.01);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        Mul8<T>(item, deviceMem, repeat, 1.01);
      });
    });
    q.wait();
  }

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mul1<T>>(gws, [=](sycl::id<1> id) {
      Mul1<T>(item, deviceMem, repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul1): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mul2<T>>(gws, [=](sycl::id<1> id) {
      Mul2<T>(item, deviceMem, repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul2): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mul4<T>>(gws, [=](sycl::id<1> id) {
      Mul4<T>(item, deviceMem, repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul4): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mul8<T>>(gws, [=](sycl::id<1> id) {
      Mul8<T>(item, deviceMem, repeat, 1.01);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MAdd1<T>(item, deviceMem, repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MAdd2<T>(item, deviceMem, repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MAdd4<T>(item, deviceMem, repeat, 10.0, 0.9899);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MAdd8<T>(item, deviceMem, repeat, 10.0, 0.9899);
      });
    });
    q.wait();
  }

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class madd1<T>>(gws, [=](sycl::id<1> id) {
      MAdd1<T>(item, deviceMem, repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd1): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class madd2<T>>(gws, [=](sycl::id<1> id) {
      MAdd2<T>(item, deviceMem, repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd2): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class madd4<T>>(gws, [=](sycl::id<1> id) {
      MAdd4<T>(item, deviceMem, repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd4): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class madd8<T>>(gws, [=](sycl::id<1> id) {
      MAdd8<T>(item, deviceMem, repeat, 10.0, 0.9899);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd8): %f (s)\n", (k_time * 1e-9f));

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MulMAdd1<T>(item, deviceMem, repeat, 3.75, 0.355);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MulMAdd2<T>(item, deviceMem, repeat, 3.75, 0.355);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MulMAdd4<T>(item, deviceMem, repeat, 3.75, 0.355);
      });
    });
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
      cgh.parallel_for<>(gws, [=](sycl::id<1> id) {
        MulMAdd8<T>(item, deviceMem, repeat, 3.75, 0.355);
      });
    });
    q.wait();
  }

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mmadd1<T>>(gws, [=](sycl::id<1> id) {
      MulMAdd1<T>(item, deviceMem, repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd1): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mmadd2<T>>(gws, [=](sycl::id<1> id) {
      MulMAdd2<T>(item, deviceMem, repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd2): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mmadd4<T>>(gws, [=](sycl::id<1> id) {
      MulMAdd4<T>(item, deviceMem, repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd4): %f (s)\n", (k_time * 1e-9f));

  //q.memcpy(&deviceMem, hostMem, sizeof(T) * numFloats).wait();
  {
    sycl::host_accessor<T, 1, sycl::access_mode::read_write> host_acc(deviceMem_buffer);
    for (size_t i = 0; i < numFloats; i++) {
    host_acc[i] = hostMem[i];  // Safe way to copy data
    }
  }
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read_write> deviceMem(deviceMem_buffer, cgh);
    cgh.parallel_for<class mmadd8<T>>(gws, [=](sycl::id<1> id) {
      MulMAdd8<T>(item, deviceMem, repeat, 3.75, 0.355);
    });
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd8): %f (s)\n", (k_time * 1e-9f));

  /*
  free(hostMem);
  sycl::free(&deviceMem, q);
  */
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  // the number of loop iterations inside kernels
  const int repeat = atoi(argv[1]);

  // a multiple of BLOCK_SIZE
  const int numFloats = 2*1024*1024;

  sycl::queue q;

  printf("=== Single-precision floating-point kernels ===\n");
  test<float>(q, repeat, numFloats);

  // comment out when double-precision is not supported by a device
  printf("=== Double-precision floating-point kernels ===\n");
  test<double>(q, repeat, numFloats);

  return 0;
}
