#ifndef MCM_MCM_CUH
#define MCM_MCM_CUH

#include "util.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__host__
void mcm_CUDA
        (float    ht,         /* time step size, 0 < ht <= 0.25 */
         int     nx,         /* image dimension in x direction */
         int     ny,         /* image dimension in y direction */
         int     nz,         /* image dimension in z direction */
         float    hx,         /* pixel width in x direction */
         float    hy,         /* pixel width in y direction */
         float    hz,         /* pixel width in y direction */
         float    *d_u);      /* input: original image ;  output: smoothed */

__host__
void mcm_iterate_CUDA(float *d_u,
                      long pmax,
                      float ht,
                      int nx, int ny, int nz,
                      float hx, float hy, float hz,
                      Npp8u *d_minmax,
                      Npp8u *d_meastd,
                      int verbose);
#ifdef __cplusplus
}
#endif

#endif //MCM_MCM_CUH