#ifndef MCM_GEODESIC_CUH
#define MCM_GEODESIC_CUH

#include "util.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void segm_geo_cuda
        (long nx,        /* image dimension in x direction */
         long ny,        /* image dimension in y direction */
         long nz,        /* image dimension in z direction */
         float alpha,     /* Balance*/
         float *d_g,      /* velocity */
         float *d_u,      /* input: original image ;  output: smoothed */
         float *d_grdx,
         float *d_grdy,
         float *d_grdz);

void segm_iterate_trace_cuda
        (float *d_g,
         float *d_u,
         float *h_u,
         float *d_grdx,
         float *d_grdy,
         float *d_grdz,
         float *h_trace,
         int *tracelength,
         int nx, int ny, int nz,
         int x1, int x2, int x3,
         int y1, int y2, int y3,
         Npp8u *d_minmax,
         Npp8u *d_meastd,
         int maxstep,
         int verbose);

#ifdef __cplusplus
}
#endif

#endif //MCM_GEODESIC_CUH