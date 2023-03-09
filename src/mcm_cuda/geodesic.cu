#include "geodesic.cuh"



extern "C" {
    __global__
    void segm_geo_kernel(const float *fg, float *ug, const float *gg, float *grdx, float *grdy, float *grdz, int3 dim) {
        // Shared memory for this block
        __shared__ float f[10][10][10];

        // Global idx
        const int gx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        const int gy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
        const int gz = (int)(blockIdx.z * blockDim.z + threadIdx.z);

        // Threads larger than the volume+1 go.
        if (gx >= dim.x) return;
        if (gy >= dim.y) return;
        if (gz >= dim.z) return;

        // Local idx
        const unsigned int i = threadIdx.x + 1;
        const unsigned int j = threadIdx.y + 1;
        const unsigned int k = threadIdx.z + 1;

        // Incomplete block borders
        int3 blockMax = make_int3(8, 8, 8);
        if (gx == dim.x-1) blockMax.x = (int)threadIdx.x + 1;
        if (gy == dim.y-1) blockMax.y = (int)threadIdx.y + 1;
        if (gz == dim.z-1) blockMax.z = (int)threadIdx.z + 1;

        // Copy the data (direct match).
        f[i][j][k] = access_3d(fg, gx, gy, gz, dim.x, dim.y);

        // Helper
        int gxm1 = per_idx(gx - 1, dim.x);
        int gym1 = per_idx(gy - 1, dim.y);
        int gzm1 = per_idx(gz - 1, dim.z);

        int gxpb = per_idx(gx + blockMax.x, dim.x);
        int gypb = per_idx(gy + blockMax.y, dim.y);
        int gzpb = per_idx(gz + blockMax.z, dim.z);

        // Overlap x
        if (threadIdx.x < 1) {
            f[i - 1][j][k] = access_3d(fg, gxm1, gy, gz, dim.x, dim.y);
            f[i + blockMax.x][j][k] = access_3d(fg, gxpb, gy, gz, dim.x, dim.y);
        }

        // Overlap y
        if (threadIdx.y < 1) {
            f[i][j - 1][k] = access_3d(fg, gx, gym1, gz, dim.x, dim.y);
            f[i][j + blockMax.y][k] = access_3d(fg, gx, gypb, gz, dim.x, dim.y);
        }

        // Overlap z
        if (threadIdx.z < 1) {
            f[i][j][k - 1] = access_3d(fg, gx, gy, gzm1, dim.x, dim.y);
            f[i][j][k + blockMax.z] = access_3d(fg, gx, gy, gzpb, dim.x, dim.y);
        }

        // Corners xy
        if (threadIdx.x < 1 && threadIdx.y < 1) {
            f[i - 1][j - 1][k] = access_3d(fg, gxm1, gym1, gz, dim.x, dim.y);
            f[i + blockMax.x][j + blockMax.y][k] = access_3d(fg, gxpb, gypb, gz, dim.x, dim.y);
            f[i - 1][j + blockMax.y][k] = access_3d(fg, gxm1, gypb, gz, dim.x, dim.y);
            f[i + blockMax.x][j - 1][k] = access_3d(fg, gxpb, gym1, gz, dim.x, dim.y);
        }

        // Corners xz
        if (threadIdx.x < 1 && threadIdx.z < 1) {
            f[i - 1][j][k - 1] = access_3d(fg, gxm1, gy, gzm1, dim.x, dim.y);
            f[i + blockMax.x][j][k + blockMax.z] = access_3d(fg, gxpb, gy, gzpb, dim.x, dim.y);
            f[i - 1][j][k + blockMax.z] = access_3d(fg, gxm1, gy, gzpb, dim.x, dim.y);
            f[i + blockMax.x][j][k - 1] = access_3d(fg, gxpb, gy, gzm1, dim.x, dim.y);
        }

        // Corners yz
        if (threadIdx.y < 1 && threadIdx.z < 1) {
            f[i][j - 1][k - 1] = access_3d(fg, gx, gym1, gzm1, dim.x, dim.y);
            f[i][j + blockMax.y][k + blockMax.z] = access_3d(fg, gx, gypb, gzpb, dim.x, dim.y);
            f[i][j - 1][k + blockMax.z] = access_3d(fg, gx, gym1, gzpb, dim.x, dim.y);
            f[i][j + blockMax.y][k - 1] = access_3d(fg, gx, gypb, gzm1, dim.x, dim.y);
        }

        // Corners all
        if (threadIdx.x < 1 && threadIdx.y < 1 && threadIdx.z < 1) {
            f[i - 1][j - 1][k - 1] = access_3d(fg, gxm1, gym1, gzm1, dim.x, dim.y);
            f[i + blockMax.x][j + blockMax.y][k + blockMax.z] = access_3d(fg, gxpb, gypb, gzpb, dim.x, dim.y);
            f[i - 1][j + blockMax.y][k + blockMax.z] = access_3d(fg, gxm1, gypb, gzpb, dim.x, dim.y);
            f[i + blockMax.x][j - 1][k + blockMax.z] = access_3d(fg, gxpb, gym1, gzpb, dim.x, dim.y);
            f[i + blockMax.x][j + blockMax.y][k - 1] = access_3d(fg, gxpb, gypb, gzm1, dim.x, dim.y);
            f[i + blockMax.x][j - 1][k - 1] = access_3d(fg, gxpb, gym1, gzm1, dim.x, dim.y);
            f[i - 1][j + blockMax.y][k - 1] = access_3d(fg, gxm1, gypb, gzm1, dim.x, dim.y);
            f[i - 1][j - 1][k + blockMax.z] = access_3d(fg, gxm1, gym1, gzpb, dim.x, dim.y);
        }

        // Sync
        __syncthreads();

        float fm_x = f[i][j][k] - f[i - 1][j][k];
        float fp_x = f[i + 1][j][k] - f[i][j][k];
        float fm_y = f[i][j][k] - f[i][j - 1][k];
        float fp_y = f[i][j + 1][k] - f[i][j][k];
        float fm_z = f[i][j][k] - f[i][j][k - 1];
        float fp_z = f[i][j][k + 1] - f[i][j][k];

        float max_fm_x, min_fp_x, max_fm_y, min_fp_y, max_fm_z, min_fp_z;

        max_fm_x = min(fm_x, 0.f);
        min_fp_x = max(fp_x, 0.f);
        max_fm_y = min(fm_y, 0.f);
        min_fp_y = max(fp_y, 0.f);
        max_fm_z = min(fm_z, 0.f);
        min_fp_z = max(fp_z, 0.f);

        /* Level set in all directions */
        float lvset = 0.5f * sqrtf(max_fm_x * max_fm_x + min_fp_x * min_fp_x
                                   + max_fm_y * max_fm_y + min_fp_y * min_fp_y
                                   + max_fm_z * max_fm_z + min_fp_z * min_fp_z);

        float val_f = f[i][j][k];
        float val_u = val_f + access_3d(gg, gx, gy, gz, dim.x, dim.y) * lvset;
        access_3d(ug, gx, gy, gz, dim.x, dim.y) = val_u;

        //u[i][j][k] = f[i][j][k] + g[i][j][k] * lvset;


        if (val_u > 0.2 && val_f < 0.2) {
            float f0_x = (f[i + 1][j][k] - f[i - 1][j][k]) / 2;
            float f0_y = (f[i][j + 1][k] - f[i][j - 1][k]) / 2;
            float f0_z = (f[i][j][k + 1] - f[i][j][k - 1]) / 2;
            float grad_sqr = sqrt(f0_x * f0_x + f0_y * f0_y + f0_z * f0_z);
            if (grad_sqr != 0.0) {
                access_3d(grdx, gx, gy, gz, dim.x, dim.y) = f0_x / grad_sqr;
                access_3d(grdy, gx, gy, gz, dim.x, dim.y) = f0_y / grad_sqr;
                access_3d(grdz, gx, gy, gz, dim.x, dim.y) = f0_z / grad_sqr;
            }
        }
    }


    __host__
    void segm_geo_cuda
            (long nx,        /* image dimension in x direction */
             long ny,        /* image dimension in y direction */
             long nz,        /* image dimension in z direction */
             float alpha,     /* Balance*/
             float *d_g,      /* velocity */
             float *d_u,      /* input: original image ;  output: smoothed */
             float *d_grdx,
             float *d_grdy,
             float *d_grdz) {
        float *d_f;                    /* u at old time level */
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
        grid.y = ny / 8 + 1;
        grid.z = nz / 8 + 1;

        int3 dim;
        dim.x = nx;
        dim.y = ny;
        dim.z = nz;

        segm_geo_kernel<<<grid, block, 1000 * sizeof(float)>>>(d_f, d_u, d_g, d_grdx, d_grdy, d_grdz, dim);

        /* ---- disallocate storage for f ---- */
        cudaFree(d_f);
    }

    __host__
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
             int verbose) {
        float alpha;                          /* Balance */
        float gc1, gc2, gc3;                  /* step width in x, y, z */
        float y1f, y2f, y3f;                  /* exact path coordinates */
        long ipx, ipy, ipz;                   /* coordinates rounded to lower integer */
        float vx1, vx2;                       /* distances from real to rounded x-coordinate, value: 0 - 1 */
        float vy1, vy2;                       /* same with y */
        float vz1, vz2;                       /* same with z */
        float h_min, h_max, h_mean, h_std;

        float *h_grdx, *h_grdy, *h_grdz;
        size_t size = nx * ny * nz;
        size_t size_bytes = size * sizeof(float);

        cudaMallocHost((void **) &h_grdx, size_bytes);
        cudaMallocHost((void **) &h_grdy, size_bytes);
        cudaMallocHost((void **) &h_grdz, size_bytes);

        if (verbose) {
            analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
            printf("Input Data: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n\n", h_min, h_max, h_mean,
                   h_std * h_std);
        }

        float h_val_u = 0.f;
        int c = 0;
        while (h_val_u < 0.2f) {
            /* perform one iteration */
            segm_geo_cuda(nx, ny, nz, alpha, d_g, d_u, d_grdx, d_grdy, d_grdz);
            c++;

            /* check minimum, maximum, mean, variance */
            if (verbose) {
                analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
                printf("LevelSet iter %d: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n", c, h_min, h_max, h_mean,
                       h_std * h_std);
            }

            /* pull testvalue from device */
            cudaMemcpy(&h_val_u, &access_3d(d_u, y1, y2, y3, nx, ny), sizeof(float), cudaMemcpyDeviceToHost);
        }

        /* pull arrays from device */
        cudaMemcpy(h_grdx, d_grdx, size_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_grdy, d_grdy, size_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_grdz, d_grdz, size_bytes, cudaMemcpyDeviceToHost);

        /* rest is CPU */
        // Null u and set start point
        memset(h_u, 0, size_bytes);
        access_3d(h_u, y1, y2, y3, nx, ny) = 1.f;
        y1f = (float) y1;
        y2f = (float) y2;
        y3f = (float) y3;

        if (verbose) {
            printf("%f, %f, %f  - %d %d %d\n", y1f + 1, y2f + 1, y3f + 1, y1 + 1, y2 + 1, y3 + 1);
            fflush(stdout);
        }

        c = 0;
        access_3d(h_trace, 0, c, 0, 3, maxstep) = y1f + 1;
        access_3d(h_trace, 1, c, 0, 3, maxstep) = y2f + 1;
        access_3d(h_trace, 2, c, 0, 3, maxstep) = y3f + 1;

        while (x1 != roundAF(y1f) || x2 != roundAF(y2f) || x3 != roundAF(y3f)) {
            /* calulate exact stepwidth in each direction for next step */
            gc1 = interpol_linear(h_grdx, y1f, y2f, y3f, nx, ny);
            gc2 = interpol_linear(h_grdy, y1f, y2f, y3f, nx, ny);
            gc3 = interpol_linear(h_grdz, y1f, y2f, y3f, nx, ny);

            /* calulate exact coordinates of next step */
            y1f = y1f + gc1;
            y2f = y2f + gc2;
            y3f = y3f + gc3;

            /* calulate shares of neighbor positions in each direction */
            ipx = (long) y1f;
            vx2 = y1f - (float) ipx;
            vx1 = 1 - vx2;
            ipy = (long) y2f;
            vy2 = y2f - (float) ipy;
            vy1 = 1 - vy2;
            ipz = (long) y3f;
            vz2 = y3f - (float) ipz;
            vz1 = 1 - vz2;

            /* calulate grey values of 8 neighbor pixels  */
            access_3d(h_u, ipx, ipy, ipz, nx, ny) += (vx1 * vy1 * vz1);
            access_3d(h_u, ipx + 1, ipy, ipz, nx, ny) += (vx2 * vy1 * vz1);
            access_3d(h_u, ipx, ipy + 1, ipz, nx, ny) += (vx1 * vy2 * vz1);
            access_3d(h_u, ipx + 1, ipy + 1, ipz, nx, ny) += (vx2 * vy2 * vz1);
            access_3d(h_u, ipx, ipy, ipz + 1, nx, ny) += (vx1 * vy1 * vz2);
            access_3d(h_u, ipx + 1, ipy, ipz + 1, nx, ny) += (vx2 * vy1 * vz2);
            access_3d(h_u, ipx, ipy + 1, ipz + 1, nx, ny) += (vx1 * vy2 * vz2);
            access_3d(h_u, ipx + 1, ipy + 1, ipz + 1, nx, ny) += (vx2 * vy2 * vz2);

            c++;
            access_3d(h_trace, 0, c, 0, 3, maxstep) = y1f + 1;
            access_3d(h_trace, 1, c, 0, 3, maxstep) = y2f + 1;
            access_3d(h_trace, 2, c, 0, 3, maxstep) = y3f + 1;

            if (verbose) {
                printf("Trace Step %d: %f, %f, %f\n", c, y1f + 1, y2f + 1, y3f + 1);
                fflush(stdout);
            }

            if (c == maxstep - 1) {
                break;
            }
        }

        if (verbose) {
            printf("Terminating after taking %d steps.\n", c);
        }

        tracelength[0] = c + 1;

        cudaFree(h_grdx);
        cudaFree(h_grdy);
        cudaFree(h_grdz);
    }
}