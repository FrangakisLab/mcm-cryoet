#include "mcm.cuh"

extern "C" {
    __global__
    void mcm_kernel(const float *fg, float *ug, float ht, float3 h, dim3 dim){
        // Shared memory for this block
        __shared__ float f[10][10][10];

        // Global idx
        const unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int gz = blockIdx.z * blockDim.z + threadIdx.z;

        if (gx >= dim.x) return;
        if (gy >= dim.y) return;
        if (gz >= dim.z) return;

        // Local idx
        const unsigned int i = threadIdx.x+1;
        const unsigned int j = threadIdx.y+1;
        const unsigned int k = threadIdx.z+1;

        // Copy the data (direct match)
        f[i][j][k] = access_3d(fg, gx, gy, gz, dim.x, dim.y);

        // Overlap x
        if (threadIdx.x < 1){
            //printf("hit %i\n", per_idx(0-1, dim.x));
            f[i-1][j][k] = access_3d(fg, per_idx(gx-1, dim.x), gy, gz, dim.x, dim.y);
            f[i+blockDim.x][j][k] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), gy, gz, dim.x, dim.y);
        }

        // Overlap y
        if (threadIdx.y < 1){
            f[i][j-1][k] = access_3d(fg, gx, per_idx(gy-1, dim.y), gz, dim.x, dim.y);
            f[i][j+blockDim.y][k] = access_3d(fg, gx, per_idx(gy+blockDim.y, dim.y), gz, dim.x, dim.y);
        }

        // Overlap z
        if (threadIdx.z < 1){
            f[i][j][k-1] = access_3d(fg, gx, gy, per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i][j][k+blockDim.z] = access_3d(fg, gx, gy, per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
        }

        // Corners xy
        if (threadIdx.x < 1 && threadIdx.y < 1){
            f[i-1][j-1][k] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy-1, dim.y), gz, dim.x, dim.y);
            f[i+blockDim.x][j+blockDim.y][k] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy+blockDim.y, dim.y), gz, dim.x, dim.y);
            f[i-1][j+blockDim.y][k] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy+blockDim.y, dim.y), gz, dim.x, dim.y);
            f[i+blockDim.x][j-1][k] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy-1, dim.y), gz, dim.x, dim.y);
        }

        // Corners xz
        if (threadIdx.x < 1 && threadIdx.z < 1){
            f[i-1][j][k-1] = access_3d(fg, per_idx(gx-1, dim.x), gy, per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j][k+blockDim.z] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), gy, per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i-1][j][k+blockDim.z] = access_3d(fg, per_idx(gx-1, dim.x), gy, per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j][k-1] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), gy, per_idx(gz-1, dim.z), dim.x, dim.y);
        }

        // Corners yz
        if (threadIdx.y < 1 && threadIdx.z < 1){
            f[i][j-1][k-1] = access_3d(fg, gx, per_idx(gy-1, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i][j+blockDim.y][k+blockDim.z] = access_3d(fg, gx, per_idx(gy+blockDim.y, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i][j-1][k+blockDim.z] = access_3d(fg, gx, per_idx(gy-1, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i][j+blockDim.y][k-1] = access_3d(fg, gx, per_idx(gy+blockDim.y, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
        }

        // Corners all
        if (threadIdx.x < 1 && threadIdx.y < 1 && threadIdx.z < 1){
            f[i-1][j-1][k-1] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy-1, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j+blockDim.y][k+blockDim.z] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy+blockDim.y, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i-1][j+blockDim.y][k+blockDim.z] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy+blockDim.y, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j-1][k+blockDim.z] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy-1, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j+blockDim.y][k-1] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy+blockDim.y, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i+blockDim.x][j-1][k-1] = access_3d(fg, per_idx(gx+blockDim.x, dim.x), per_idx(gy-1, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i-1][j+blockDim.y][k-1] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy+blockDim.y, dim.y), per_idx(gz-1, dim.z), dim.x, dim.y);
            f[i-1][j-1][k+blockDim.z] = access_3d(fg, per_idx(gx-1, dim.x), per_idx(gy-1, dim.y), per_idx(gz+blockDim.z, dim.z), dim.x, dim.y);
        }

        // Sync
        __syncthreads();

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
        if (grad_sqr != 0.0) {
            access_3d(ug, gx, gy, gz, dim.x, dim.y) = f[i][j][k]
                                                      + ht * (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) +
                                                              fz * fz * (fxx + fyy)
                                                              - 2.0f * fx * fy * fxy - 2.0f * fy * fz * fyz -
                                                              2.0f * fx * fz * fxz) / grad_sqr;
        }

    }

    __host__
    void mcm_CUDA
            (float    ht,        /* time step size, 0 < ht <= 0.25 */
             long     nx,        /* image dimension in x direction */
             long     ny,        /* image dimension in y direction */
             long     nz,        /* image dimension in z direction */
             float    hx,        /* pixel width in x direction */
             float    hy,        /* pixel width in y direction */
             float    hz,        /* pixel width in y direction */
             float    *d_u)      /* input: original image ;  output: smoothed */

    {
        float *d_f;
        size_t size = nx * ny * nz;
        size_t size_bytes = size * sizeof(float);

    /* ---- allocate storage f ---- */
        cudaMalloc((void **) &d_f, size_bytes);

    /* ---- copy u into f ---- */
        cudaMemcpy(d_f, d_u, size_bytes, cudaMemcpyDeviceToDevice);

    /* loop */
        dim3 block;
        block.x = 8;
        block.y = 8;
        block.z = 8;

        dim3 grid;
        grid.x = nx / 8 + 1;
        grid.y = nx / 8 + 1;
        grid.z = nx / 8 + 1;

        dim3 dim;
        dim.x = nx;
        dim.y = ny;
        dim.z = nz;

        float3 h;
        h.x = hx;
        h.y = hy;
        h.z = hz;

        mcm_kernel<<<grid, block, 1000 * sizeof(float)>>>(d_f, d_u, ht, h, dim);

    /* ---- disallocate storage for f ---- */
        cudaFree(d_f);
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
