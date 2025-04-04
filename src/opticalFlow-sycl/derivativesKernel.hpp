#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
void ComputeDerivativesKernel(int width, int height, int stride, float *Ix,
                              float *Iy, float *Iz,
                              sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> texSource,
                              sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> texTarget,
                              sycl::sampler texDesc,
                              const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float t0, t1;

  auto x_inputCoords1 = sycl::float2(ix - 2.0f, iy);
  auto x_inputCoords2 = sycl::float2(ix - 1.0f, iy);
  auto x_inputCoords3 = sycl::float2(ix + 1.0f, iy);
  auto x_inputCoords4 = sycl::float2(ix + 2.0f, iy);

  t0 = texSource.read(x_inputCoords1, texDesc)[0];
  t0 -= texSource.read(x_inputCoords2, texDesc)[0] * 8.0f;
  t0 += texSource.read(x_inputCoords3, texDesc)[0] * 8.0f;
  t0 -= texSource.read(x_inputCoords4, texDesc)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x_inputCoords1, texDesc)[0];
  t1 -= texTarget.read(x_inputCoords2, texDesc)[0] * 8.0f;
  t1 += texTarget.read(x_inputCoords3, texDesc)[0] * 8.0f;
  t1 -= texTarget.read(x_inputCoords4, texDesc)[0];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  auto inputCoord = sycl::float2(ix, iy);
  Iz[pos] = texTarget.read(inputCoord, texDesc)[0] -
            texSource.read(inputCoord, texDesc)[0];

  // y derivative
  auto y_inputCoords1 = sycl::float2(ix, iy - 2.0f);
  auto y_inputCoords2 = sycl::float2(ix, iy - 1.0f);
  auto y_inputCoords3 = sycl::float2(ix, iy + 1.0f);
  auto y_inputCoords4 = sycl::float2(ix, iy + 2.0f);

  t0 = texSource.read(y_inputCoords1, texDesc)[0];
  t0 -= texSource.read(y_inputCoords2, texDesc)[0] * 8.0f;
  t0 += texSource.read(y_inputCoords3, texDesc)[0] * 8.0f;
  t0 -= texSource.read(y_inputCoords4, texDesc)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(y_inputCoords1, texDesc)[0];
  t1 -= texTarget.read(y_inputCoords2, texDesc)[0] * 8.0f;
  t1 += texTarget.read(y_inputCoords3, texDesc)[0] * 8.0f;
  t1 -= texTarget.read(y_inputCoords4, texDesc)[0];
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, float *pI0_h,
                               float *pI1_h, float *I0_h, float *I1_h,
                               float *src_d0, float *src_d1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz, sycl::queue q) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  int dataSize = s * h * sizeof(float);

  q.memcpy(I0_h, I0, dataSize);
  q.memcpy(I1_h, I1, dataSize);
  q.wait();

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      pI0_h[index * 4 + 0] = I0_h[index];
      pI0_h[index * 4 + 1] = pI0_h[index * 4 + 2] = pI0_h[index * 4 + 3] = 0.f;
    }
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int index = i * s + j;
      pI1_h[index * 4 + 0] = I1_h[index];
      pI1_h[index * 4 + 1] = pI1_h[index * 4 + 2] = pI1_h[index * 4 + 3] = 0.f;
    }
  }

  q.memcpy(src_d0, pI0_h, s * h * sizeof(sycl::float4));
  q.memcpy(src_d1, pI1_h, s * h * sizeof(sycl::float4));

  q.wait();

  auto texDescr = sycl::sampler(
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest);

  auto texSource =
      sycl::image<2>(src_d0, sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32, sycl::range<2>(w, h),
                         sycl::range<1>(s * sizeof(sycl::float4)));

  auto texTarget =
      sycl::image<2>(src_d1, sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32, sycl::range<2>(w, h),
                         sycl::range<1>(s * sizeof(sycl::float4)));
  
  
  q.submit([&](sycl::handler &cgh) {
    auto texSource_acc =
         texSource.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);
     auto texTarget_acc =
         texTarget.template get_access<sycl::float4,
                                       sycl::access::mode::read>(cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item) {
          ComputeDerivativesKernel(
              w, h, s, Ix, Iy, Iz,
             texSource_acc, texTarget_acc,
              texDescr, item);
        });
  });
}
