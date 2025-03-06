main_with_nd_range.cpp:
This version does not work with OpenMP target of IRIS,
this is probably because of lack of implementation of 
ND-Range kernels in CHARM-SYCL.

main_with_range.cpp:
This version works for all targets but uses sycl::range,
instead of sycl::nd_range. Since sycl::nd_range allows
setting local work sizes not using it means, we cannot
set local work sizes.
