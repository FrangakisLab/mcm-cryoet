#ifndef MCM_LIBMCM_CUDA_H
#define MCM_LIBMCM_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void run_combi_gpu(const float* h_input,       /* input image */
                   float* output,              /* output smoothed image */
                   int iterations,             /* iterations of mcm/levelset smoothing */
                   float alpha,                /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
                   float beta,                 /* Amount of mc motion 0 < beta < 1 */
                   int nx, int ny, int nz,     /* image dimensions */
                   int verbose);               /* whether to compute and print image stats */

void run_mcm_gpu(const float* input,              /* input image */
                 float* output,                   /* output smoothed image */
                 int iterations,                  /* iterations of mcm smoothing */
                 int nx, int ny, int nz,          /* image dimensions */
                 float hx, float hy, float hz,    /* pixel width in x,y,z direction */
                 int verbose);                    /* whether to compute and print image stats */

void run_trace_gpu(const float* speed_input,  /* input speed image */
                   float* output_vol,         /* output path image */
                   float* output_trace,       /* output path coordinates, allocate 3 x maxstep+1 */
                   int* tracelength,          /* actual output path length */
                   int x1, int x2, int x3,    /* end coordinate (one-based) */
                   int y1, int y2, int y3,    /* start coordinate (one-based) */
                   int nx, int ny, int nz,    /* image dimensions */
                   int maxstep,               /* maximum path length (good default: 10000) */
                   int verbose);              /* whether to compute and print image stats */

#ifdef __cplusplus
}
#endif
#endif //MCM_LIBMCM_CUDA_H
