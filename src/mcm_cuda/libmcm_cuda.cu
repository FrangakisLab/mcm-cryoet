#include "../../include/libmcm_cuda.cuh"
#include "combi.cuh"
#include "mcm.cuh"
#include "geodesic.cuh"

extern "C"
__host__
void run_combi_gpu(const float* h_input,       /* input image */
                   float* output,              /* output smoothed image */
                   int iterations,             /* iterations of mcm/levelset smoothing */
                   float alpha,                /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
                   float beta,                 /* Amount of mc motion 0 < beta < 1 */
                   int nx, int ny, int nz,     /* image dimensions */
                   int verbose)                /* whether to compute and print image stats */
{
    float *d_u;                       /* linear device storage */

    int minmax_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_minmax;                  /* device buffer */

    int meastd_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_meastd;                  /* device buffer */

/* allocate storage */
    size_t size = nx * ny * nz;
    size_t size_bytes = size * sizeof(float);
    cudaMalloc((void**)&d_u, size_bytes);

    if (verbose){
        // Min/Max setup
        nppsMinMaxGetBufferSize_32f((int)size, &minmax_buf_size);
        cudaMalloc((void**)&d_minmax, minmax_buf_size);

        // Mean/Std setup
        nppsMeanStdDevGetBufferSize_32f((int)size, &meastd_buf_size);
        cudaMalloc((void**)&d_meastd, meastd_buf_size);
    }

    // Send to GPU
    cudaMemcpy(d_u, h_input, size_bytes, cudaMemcpyHostToDevice);

/* ---- process image ---- */
    segm_combi_iterate_CUDA(d_u, iterations, nx, ny, nz, alpha, beta, d_minmax, d_meastd, verbose);

/* ---- get output image ---- */
    cudaMemcpy(output, d_u, size_bytes, cudaMemcpyDeviceToHost);

/* ---- disallocate storage ---- */
    cudaFree(d_u);

    if(verbose) {
        cudaFree(d_minmax);
        cudaFree(d_meastd);
    }
}

extern "C"
__host__
void run_mcm_gpu(const float* input,              /* input image */
                 float* output,                   /* output smoothed image */
                 int iterations,                  /* iterations of mcm smoothing */
                 int nx, int ny, int nz,          /* image dimensions */
                 float hx, float hy, float hz,    /* pixel width in x,y,z direction */
                 int verbose)                     /* whether to compute and print image stats */
{
    float *d_u;                       /* linear device storage */

    int minmax_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_minmax;                  /* device buffer */

    int meastd_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_meastd;                  /* device buffer */

/* ---- read other parameters ---- */
    float ht=0.2f;

/* allocate storage */
    size_t size = nx * ny * nz;
    size_t size_bytes = size * sizeof(float);
    cudaMalloc((void**)&d_u, size_bytes);

    if(verbose) {
        // Min/Max setup
        nppsMinMaxGetBufferSize_32f((int) size, &minmax_buf_size);
        cudaMalloc((void **) &d_minmax, minmax_buf_size);

        // Mean/Std setup
        nppsMeanStdDevGetBufferSize_32f((int) size, &meastd_buf_size);
        cudaMalloc((void **) &d_meastd, meastd_buf_size);
    }

    // Send to GPU
    cudaMemcpy(d_u, input, size_bytes, cudaMemcpyHostToDevice);

/* ---- process image ---- */
    mcm_iterate_CUDA(d_u, iterations, ht, nx, ny, nz, hx, hy, hz, d_minmax, d_meastd, verbose);

/* ---- get output image ---- */
    cudaMemcpy(output, d_u, size_bytes, cudaMemcpyDeviceToHost);

/* ---- disallocate storage ---- */
    cudaFree(d_u);

    if(verbose) {
        cudaFree(d_minmax);
        cudaFree(d_meastd);
    }
}


extern "C"
__host__
void run_trace_gpu(const float* speed_input,  /* input speed image */
                   float* output_vol,         /* output path image */
                   float* output_trace,       /* output path coordinates, allocate 3 x maxstep+1 */
                   int* tracelength,          /* actual output path length */
                   int x1, int x2, int x3,    /* end coordinate (one-based) */
                   int y1, int y2, int y3,    /* start coordinate (one-based) */
                   int nx, int ny, int nz,    /* image dimensions */
                   int maxstep,               /* maximum path length (good default: 10000) */
                   int verbose)               /* whether to compute and print image stats */

/* Computes the shortest geodesic path from start point to end point through mask. */
{
    float *d_u;                       /* linear device storage image*/

    float *d_g;                       /* linear device storage speed*/

    float *d_grdx;                    /* linear device storage gradient x*/
    float *d_grdy;                    /* linear device storage gradient z*/
    float *d_grdz;                    /* linear device storage gradient z*/

    int minmax_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_minmax;                  /* device buffer */

    int meastd_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_meastd;                  /* device buffer */

/* adjust coords */
    x1 = x1 - 1;
    x2 = x2 - 1;
    x3 = x3 - 1;

    y1 = y1 - 1;
    y2 = y2 - 1;
    y3 = y3 - 1;

/* allocate storage */
    size_t size = nx * ny * nz;
    size_t size_bytes = size * sizeof(float);

    cudaMalloc((void**)&d_u, size_bytes);

    cudaMalloc((void**)&d_g, size_bytes);

    cudaMalloc((void**)&d_grdx, size_bytes);
    cudaMalloc((void**)&d_grdy, size_bytes);
    cudaMalloc((void**)&d_grdz, size_bytes);

    cudaMemset(d_grdx, 0, size_bytes);
    cudaMemset(d_grdy, 0, size_bytes);
    cudaMemset(d_grdz, 0, size_bytes);

/* Initiate levelset with zeros */
    cudaMemset(d_u, 0, size_bytes);
    float one = 1.f;

/*Set start position of level set to 1 */
    cudaMemcpy(&access_3d(d_u, x1, x2, x3, nx, ny), &one, sizeof(float), cudaMemcpyHostToDevice);

/*Send speed file to device */
    cudaMemcpy(d_g, speed_input, size_bytes, cudaMemcpyHostToDevice);
    if (access_3d(speed_input, x1, x2, x3, nx, ny)<0.9 || access_3d(speed_input, y1, y2, y3, nx, ny)<0.9 )
    {
        printf("Error: Pixel values not equal to 1.\n");
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", x1+1, x2+1, x3+1, access_3d(speed_input, x1+1, x2+1, x3+1, nx, ny));
        printf("Value at y1=%d, y2=%d, y3=%d : %f\n", y1+1, y2+1, y3+1, access_3d(speed_input, y1+1, y2+1, y3+1, nx, ny));
        printf("exiting program\n");
        exit(1);
    }

    if(verbose) {
        // Min/Max setup
        nppsMinMaxGetBufferSize_32f((int) size, &minmax_buf_size);
        cudaMalloc((void **) &d_minmax, minmax_buf_size);

        // Mean/Std setup
        nppsMeanStdDevGetBufferSize_32f((int) size, &meastd_buf_size);
        cudaMalloc((void **) &d_meastd, meastd_buf_size);
    }



/* ---- process image ---- */
    segm_iterate_trace_cuda(d_g, d_u, output_vol, d_grdx, d_grdy, d_grdz,
                            output_trace, tracelength,
                            nx, ny, nz,
                            x1, x2, x3,
                            y1, y2, y3,
                            d_minmax,
                            d_meastd,
                            maxstep,
                            verbose);

/* ---- get output image ---- */
    // --> results already in outputs

/* ---- disallocate storage ---- */
    cudaFree(d_u);
    cudaFree(d_g);
    cudaFree(d_grdx);
    cudaFree(d_grdy);
    cudaFree(d_grdz);
    cudaFree(d_minmax);
    cudaFree(d_meastd);
}