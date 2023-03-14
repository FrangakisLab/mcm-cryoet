#include "mcm.cuh"

extern "C" {
    __global__
    void mcm_kernel(const float *fg, float *ug, float ht, float3 h, int3 dim){
        // Shared memory for this block
        __shared__ float f[10][10][10];

        // Global idx
        const int3 g = make_int3((int)(blockIdx.x * blockDim.x + threadIdx.x),
                                 (int)(blockIdx.y * blockDim.y + threadIdx.y),
                                 (int)(blockIdx.z * blockDim.z + threadIdx.z));

        // Local idx
        const int3 l = make_int3((int)threadIdx.x + 1,
                                 (int)threadIdx.y + 1,
                                 (int)threadIdx.z + 1);

        int i = (int) threadIdx.x + 1;
        int j = (int) threadIdx.y + 1;
        int k = (int) threadIdx.z + 1;

        if (g.x < dim.x & g.y < dim.y & g.z < dim.z) {
            copy_stencil(f, fg, dim, g, l);
        }

        // Sync
        __syncthreads();

        // Threads larger than the volume go now that we synced.
        if (g.x >= dim.x) return;
        if (g.y >= dim.y) return;
        if (g.z >= dim.z) return;

        /* calculate some time savers */
        float3 two_h, h_sqr;

        two_h.x = 2.0f * h.x;
        two_h.y = 2.0f * h.y;
        two_h.z = 2.0f * h.z;
        h_sqr.x = h.x * h.x;
        h_sqr.y = h.y * h.y;
        h_sqr.z = h.z * h.z;
        float two_hx_hy = 2.0f * h.x * h.y;
        float two_hx_hz = 2.0f * h.x * h.z;
        float two_hy_hz = 2.0f * h.y * h.z;

        /* central spatial derivatives */
        float fx  = (f[i+1][j][k] - f[i-1][j][k]) / two_h.x;
        float fy  = (f[i][j+1][k] - f[i][j-1][k]) / two_h.y;
        float fz  = (f[i][j][k+1] - f[i][j][k-1]) / two_h.z;
        float fxx = (f[i+1][j][k] - 2.0f * f[i][j][k] + f[i-1][j][k]) / h_sqr.x;
        float fyy = (f[i][j+1][k] - 2.0f * f[i][j][k] + f[i][j-1][k]) / h_sqr.y;
        float fzz = (f[i][j][k+1] - 2.0f * f[i][j][k] + f[i][j][k-1]) / h_sqr.z;

        float fxy, fyz, fxz;

        if (fx * fy < 0.0f)
            fxy = (   f[i+1][j+1][k] - f[i][j+1][k] - f[i+1][j][k] + f[i][j][k]
                      + f[i-1][j-1][k] - f[i][j-1][k] - f[i-1][j][k] + f[i][j][k] )
                  / two_hx_hy;
        else
            fxy = ( - f[i-1][j+1][k] + f[i][j+1][k] + f[i+1][j][k] - f[i][j][k]
                    - f[i+1][j-1][k] + f[i][j-1][k] + f[i-1][j][k] - f[i][j][k] )
                  / two_hx_hy;

        if (fy * fz < 0.0f)
            fyz = (   f[i][j+1][k+1] - f[i][j+1][k] - f[i][j][k+1] + f[i][j][k]
                      + f[i][j-1][k-1] - f[i][j-1][k] - f[i][j][k-1] + f[i][j][k] )
                  / two_hy_hz;
        else
            fyz = ( - f[i][j+1][k-1] + f[i][j+1][k] + f[i][j][k+1] - f[i][j][k]
                    - f[i][j-1][k+1] + f[i][j-1][k] + f[i][j][k-1] - f[i][j][k] )
                  / two_hy_hz;

        if (fx * fz < 0.0f)
            fxz = (   f[i+1][j][k+1] - f[i+1][j][k] - f[i][j][k+1] + f[i][j][k]
                      + f[i-1][j][k-1] - f[i-1][j][k] - f[i][j][k-1] + f[i][j][k] )
                  / two_hx_hz;
        else
            fxz = ( - f[i-1][j][k+1] + f[i][j][k+1] + f[i+1][j][k] - f[i][j][k]
                    - f[i+1][j][k-1] + f[i][j][k-1] + f[i-1][j][k] - f[i][j][k] )
                  / two_hx_hz;



        float grad_sqr = (fx * fx + fy * fy + fz * fz);
        if (grad_sqr != 0.0f) {
            access_3d(ug, g.x, g.y, g.z, dim.x, dim.y) = f[i][j][k]
                                                      + ht * (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) +
                                                              fz * fz * (fxx + fyy)
                                                              - 2.0f * fx * fy * fxy - 2.0f * fy * fz * fyz -
                                                              2.0f * fx * fz * fxz) / grad_sqr;
        }

    }

    __host__
    void mcm_CUDA
            (float    ht,        /* time step size, 0 < ht <= 0.25 */
             int      nx,        /* image dimension in x direction */
             int      ny,        /* image dimension in y direction */
             int      nz,        /* image dimension in z direction */
             float    hx,        /* pixel width in x direction */
             float    hy,        /* pixel width in y direction */
             float    hz,        /* pixel width in y direction */
             float    *d_u)      /* input: original image ;  output: smoothed */

    {
        float *d_f;
        size_t size = nx * ny * nz;
        size_t size_bytes = size * sizeof(float);

    /* ---- allocate storage f ---- */
        CUDA_CALL(cudaMalloc((void **) &d_f, size_bytes));

    /* ---- copy u into f ---- */
        CUDA_CALL(cudaMemcpy(d_f, d_u, size_bytes, cudaMemcpyDeviceToDevice));

    /* loop */
        dim3 block;
        block.x = 8;
        block.y = 8;
        block.z = 8;

        dim3 grid;
        grid.x = nx / 8 + 1;
        grid.y = ny / 8 + 1;
        grid.z = nz / 8 + 1;

        int3 dim;
        dim.x = nx;
        dim.y = ny;
        dim.z = nz;

        float3 h;
        h.x = hx;
        h.y = hy;
        h.z = hz;

        mcm_kernel<<<grid, block>>>(d_f, d_u, ht, h, dim);

    /* ---- disallocate storage for f ---- */
        CUDA_CALL(cudaFree(d_f));
    }

    __host__
    void mcm_iterate_CUDA(float *d_u,
                          long pmax,
                          float ht,
                          int nx, int ny, int nz,
                          float hx, float hy, float hz,
                          Npp8u *d_minmax,
                          Npp8u *d_meastd,
                          int verbose)
    {
        float h_min, h_max, h_mean, h_std;

        if (verbose) {
            analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
            printf("Input Data: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n\n", h_min, h_max, h_mean, h_std * h_std);
        }

        for (long p=1; p<=pmax; p++) {
            /* perform one iteration */
            if(verbose) {
                printf("iteration number: %5ld / %ld \n", p, pmax);
            }

            mcm_CUDA(ht, nx, ny, nz, hx, hy, hz, d_u);

            /* check minimum, maximum, mean, variance */
            if (verbose) {
                analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
                printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", h_min, h_max, h_mean, h_std*h_std);
            }
        }
    }

}
