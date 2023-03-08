#ifndef MCM_LEVELSET_COMBI_H
#define MCM_LEVELSET_COMBI_H

#include "util.cuh"

#ifdef __cplusplus
extern "C" {
#endif
__host__
void segm_combi_CUDA
        (int nx,        /* image dimension in x direction */
         int ny,        /* image dimension in y direction */
         int nz,        /* image dimension in z direction */
         float alpha,      /* Balance*/
         float beta,
         float *d_u);       /* input: original image ;  output: smoothed */

void segm_combi_iterate_CUDA(float *d_u,
                             int pmax,
                             int nx, int ny, int nz,
                             float alpha,
                             float beta,
                             Npp8u *d_minmax,
                             Npp8u *d_meastd,
                             int verbose);
#ifdef __cplusplus
}
#endif

#endif //MCM_LEVELSET_COMBI_H