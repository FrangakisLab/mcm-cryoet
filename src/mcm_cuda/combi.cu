#include "combi.cuh"

extern "C" {
    __global__
    void segm_combi_kernel(const float *fg, float *ug, float alpha, float beta, dim3 dim)
    {
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

        // Compute curvature
        float f0_x = (f[i + 1][j][k] - f[i - 1][j][k]) / 2.f;
        float f0_y = (f[i][j + 1][k] - f[i][j - 1][k]) / 2.f;
        float f0_z = (f[i][j][k + 1] - f[i][j][k - 1]) / 2.f;
        float f0_xx = (f[i + 1][j][k] - 2.0f * f[i][j][k] + f[i - 1][j][k]);
        float f0_yy = (f[i][j + 1][k] - 2.0f * f[i][j][k] + f[i][j - 1][k]);
        float f0_zz = (f[i][j][k + 1] - 2.0f * f[i][j][k] + f[i][j][k - 1]);

        float f0_xy, f0_yz, f0_xz;

        if (f0_x * f0_y < 0.0)
            f0_xy = (f[i + 1][j + 1][k] - f[i][j + 1][k] - f[i + 1][j][k] + f[i][j][k]
                     + f[i - 1][j - 1][k] - f[i][j - 1][k] - f[i - 1][j][k] + f[i][j][k]) / 2.f;
        else
            f0_xy = (-f[i - 1][j + 1][k] + f[i][j + 1][k] + f[i + 1][j][k] - f[i][j][k]
                     - f[i + 1][j - 1][k] + f[i][j - 1][k] + f[i - 1][j][k] - f[i][j][k]) / 2.f;

        if (f0_y * f0_z < 0.0)
            f0_yz = (f[i][j + 1][k + 1] - f[i][j + 1][k] - f[i][j][k + 1] + f[i][j][k]
                     + f[i][j - 1][k - 1] - f[i][j - 1][k] - f[i][j][k - 1] + f[i][j][k]) / 2.f;
        else
            f0_yz = (-f[i][j + 1][k - 1] + f[i][j + 1][k] + f[i][j][k + 1] - f[i][j][k]
                     - f[i][j - 1][k + 1] + f[i][j - 1][k] + f[i][j][k - 1] - f[i][j][k]) / 2.f;

        if (f0_x * f0_z < 0.0)
            f0_xz = (f[i + 1][j][k + 1] - f[i + 1][j][k] - f[i][j][k + 1] + f[i][j][k]
                     + f[i - 1][j][k - 1] - f[i - 1][j][k] - f[i][j][k - 1] + f[i][j][k]) / 2.f;
        else
            f0_xz = (-f[i - 1][j][k + 1] + f[i][j][k + 1] + f[i + 1][j][k] - f[i][j][k]
                     - f[i + 1][j][k - 1] + f[i][j][k - 1] + f[i - 1][j][k] - f[i][j][k]) / 2.f;

        float grad_sqr = f0_x * f0_x + f0_y * f0_y + f0_z * f0_z;

        float curv = 0.0f;

        if (grad_sqr != 0.0f) {
            curv =  0.2f * (f0_x * f0_x * (f0_yy + f0_zz) + f0_y * f0_y * (f0_xx + f0_zz)
                           + f0_z * f0_z * (f0_xx + f0_yy) - 2.0f * f0_x * f0_y * f0_xy
                           - 2.0f * f0_y * f0_z * f0_yz - 2.0f * f0_x * f0_z * f0_xz) / grad_sqr;
        }

        float fm_x = f[i][j][k] - f[i - 1][j][k];
        float fp_x = f[i + 1][j][k] - f[i][j][k];
        float fm_y = f[i][j][k] - f[i][j - 1][k];
        float fp_y = f[i][j + 1][k] - f[i][j][k];
        float fm_z = f[i][j][k] - f[i][j][k - 1];
        float fp_z = f[i][j][k + 1] - f[i][j][k];

        float max_fm_x, min_fp_x, max_fm_y, min_fp_y, max_fm_z, min_fp_z;

        if (alpha > 0.f) {
            max_fm_x = min(fm_x, 0.f);
            min_fp_x = max(fp_x, 0.f);
            max_fm_y = min(fm_y, 0.f);
            min_fp_y = max(fp_y, 0.f);
            max_fm_z = min(fm_z, 0.f);
            min_fp_z = max(fp_z, 0.f);
        } else {
            max_fm_x = max(fm_x, 0.f);
            min_fp_x = min(fp_x, 0.f);
            max_fm_y = max(fm_y, 0.f);
            min_fp_y = min(fp_y, 0.f);
            max_fm_z = max(fm_z, 0.f);
            min_fp_z = min(fp_z, 0.f);
        }

        /* Level set in all directions */
        float lvset = 0.5f * sqrtf(max_fm_x * max_fm_x + min_fp_x * min_fp_x
                                   + max_fm_y * max_fm_y + min_fp_y * min_fp_y
                                   + max_fm_z * max_fm_z + min_fp_z * min_fp_z);

        // Return
        access_3d(ug, gx, gy, gz, dim.x, dim.y) = f[i][j][k] + beta * curv + alpha * lvset;
    }

    __host__
    void segm_combi_CUDA

            (int nx,        /* image dimension in x direction */
             int ny,        /* image dimension in y direction */
             int nz,        /* image dimension in z direction */
             float alpha,      /* Balance*/
             float beta,
             float *d_u)       /* input: original image ;  output: smoothed */

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

        segm_combi_kernel<<<grid, block, 1000 * sizeof(float)>>>(d_f, d_u, alpha, beta, dim);

/* ---- disallocate storage for f ---- */
        cudaFree(d_f);
    }

    __host__
    void segm_combi_iterate_CUDA(float *d_u,
                                 int pmax,
                                 int nx, int ny, int nz,
                                 float alpha,
                                 float beta,
                                 Npp8u *d_minmax,
                                 Npp8u *d_meastd,
                                 int verbose)
    {
        float h_min, h_max, h_mean, h_std;

        if (verbose) {
            analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
            printf("Input Data: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n\n", h_min, h_max, h_mean, h_std * h_std);
        }

        for (int p=1; p<=pmax; p++)
        {
            /* perform one iteration */
            if (verbose) {
                printf("iteration number: %5d / %d \n", p, pmax);
            }
            segm_combi_CUDA(nx, ny, nz, alpha, beta, d_u);

            /* check minimum, maximum, mean, variance */
            if (verbose) {
                analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
                printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", h_min, h_max, h_mean, h_std*h_std);
            }
        }
    }

}