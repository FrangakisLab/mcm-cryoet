#ifndef MCM_UTIL_CUH
#define MCM_UTIL_CUH

#include <cuda_runtime.h>
#include <npps.h>
#include <stdio.h>
#include <stdlib.h>

// Periodic boundary condition indeces
#define per_idx(idx, dim) ( ((dim) + ((idx) % (dim))) % (dim) )

// Reflected boundary condition indeces
//TODO: reflected boundary

// 3d access tp linear array
#define access_3d(arr, x, y, z, dx, dy) (arr[(x) + ((dx) * (y)) + ((dx) * (dy) * (z))])



#ifdef __cplusplus
#include <iostream>

// CUDA error checking
#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

// NPP error checking
#define NPP_CALL( call )               \
{                                       \
NppStatus result = call;              \
if ( NPP_SUCCESS != result )            \
    std::cerr << "NPP error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << result << " (" << #call << ")" << std::endl;  \
}

extern "C" {
#endif

void analyse_CUDA
        (float *d_u,
         int nx,
         int ny,
         int nz,
         Npp8u *d_buf1,
         Npp8u *d_buf2,
         float *h_min,
         float *h_max,
         float *h_mean,
         float *h_std);

float interpol_linear
        (const float   *image,
         float   x,
         float   y,
         float   z,
         int     nx,
         int     ny);

long roundAF (float value);

#ifdef __cplusplus
}
#endif

#endif //MCM_UTIL_CUH
