#include "util.cuh"

extern "C" {
    __device__
    void copy_stencil(float dst[10][10][10], const float *src, int3 dim, int3 g, int3 l) {

        // Inomplete block borders
        int3 blockMax = make_int3(8, 8, 8);
        if (blockIdx.x == gridDim.x - 1) blockMax.x = blockMax.x - ((int) ((blockIdx.x + 1) * blockDim.x) % dim.x);
        if (blockIdx.y == gridDim.y - 1) blockMax.y = blockMax.y - ((int) ((blockIdx.y + 1) * blockDim.y) % dim.y);
        if (blockIdx.z == gridDim.z - 1) blockMax.z = blockMax.z - ((int) ((blockIdx.z + 1) * blockDim.z) % dim.z);

        // Helper
        const int3 gm1 = make_int3(ext_idx(g.x - 1, dim.x),
                                   ext_idx(g.y - 1, dim.y),
                                   ext_idx(g.z - 1, dim.z));

        const int3 gpb = make_int3(ext_idx(g.x + blockMax.x, dim.x),
                                   ext_idx(g.y + blockMax.y, dim.y),
                                   ext_idx(g.z + blockMax.z, dim.z));

        // Copy the data (direct match).
        dst[l.x][l.y][l.z] = access_3d(src, g.x, g.y, g.z, dim.x, dim.y);

        // Overlap x
        if (threadIdx.x < 1) {
            dst[l.x - 1][l.y][l.z] = access_3d(src, gm1.x, g.y, g.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y][l.z] = access_3d(src, gpb.x, g.y, g.z, dim.x, dim.y);
        }

        // Overlap y
        if (threadIdx.y < 1) {
            dst[l.x][l.y - 1][l.z] = access_3d(src, g.x, gm1.y, g.z, dim.x, dim.y);
            dst[l.x][l.y + blockMax.y][l.z] = access_3d(src, g.x, gpb.y, g.z, dim.x, dim.y);
        }

        // Overlap z
        if (threadIdx.z < 1) {
            dst[l.x][l.y][l.z - 1] = access_3d(src, g.x, g.y, gm1.z, dim.x, dim.y);
            dst[l.x][l.y][l.z + blockMax.z] = access_3d(src, g.x, g.y, gpb.z, dim.x, dim.y);
        }

        // Corners xy
        if (threadIdx.x < 1 && threadIdx.y < 1) {
            dst[l.x - 1][l.y - 1][l.z] = access_3d(src, gm1.x, gm1.y, g.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y + blockMax.y][l.z] = access_3d(src, gpb.x, gpb.y, g.z, dim.x, dim.y);
            dst[l.x - 1][l.y + blockMax.y][l.z] = access_3d(src, gm1.x, gpb.y, g.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y - 1][l.z] = access_3d(src, gpb.x, gm1.y, g.z, dim.x, dim.y);
        }

        // Corners xz
        if (threadIdx.x < 1 && threadIdx.z < 1) {
            dst[l.x - 1][l.y][l.z - 1] = access_3d(src, gm1.x, g.y, gm1.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y][l.z + blockMax.z] = access_3d(src, gpb.x, g.y, gpb.z, dim.x, dim.y);
            dst[l.x - 1][l.y][l.z + blockMax.z] = access_3d(src, gm1.x, g.y, gpb.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y][l.z - 1] = access_3d(src, gpb.x, g.y, gm1.z, dim.x, dim.y);
        }

        // Corners yz
        if (threadIdx.y < 1 && threadIdx.z < 1) {
            dst[l.x][l.y - 1][l.z - 1] = access_3d(src, g.x, gm1.y, gm1.z, dim.x, dim.y);
            dst[l.x][l.y + blockMax.y][l.z + blockMax.z] = access_3d(src, g.x, gpb.y, gpb.z, dim.x, dim.y);
            dst[l.x][l.y - 1][l.z + blockMax.z] = access_3d(src, g.x, gm1.y, gpb.z, dim.x, dim.y);
            dst[l.x][l.y + blockMax.y][l.z - 1] = access_3d(src, g.x, gpb.y, gm1.z, dim.x, dim.y);
        }

        // Corners all
        if (threadIdx.x < 1 && threadIdx.y < 1 && threadIdx.z < 1) {
            dst[l.x - 1][l.y - 1][l.z - 1] = access_3d(src, gm1.x, gm1.y, gm1.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y + blockMax.y][l.z + blockMax.z] = access_3d(src, gpb.x, gpb.y, gpb.z, dim.x, dim.y);

            dst[l.x - 1][l.y + blockMax.y][l.z + blockMax.z] = access_3d(src, gm1.x, gpb.y, gpb.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y - 1][l.z + blockMax.z] = access_3d(src, gpb.x, gm1.y, gpb.z, dim.x, dim.y);
            dst[l.x + blockMax.x][l.y + blockMax.y][l.z - 1] = access_3d(src, gpb.x, gpb.y, gm1.z, dim.x, dim.y);

            dst[l.x + blockMax.x][l.y - 1][l.z - 1] = access_3d(src, gpb.x, gm1.y, gm1.z, dim.x, dim.y);
            dst[l.x - 1][l.y + blockMax.y][l.z - 1] = access_3d(src, gm1.x, gpb.y, gm1.z, dim.x, dim.y);
            dst[l.x - 1][l.y - 1][l.z + blockMax.z] = access_3d(src, gm1.x, gm1.y, gpb.z, dim.x, dim.y);
        }
    }


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
             float *h_std) {
        // Size in voxels
        size_t size = nx * ny * nz;

        // Alloc
        Npp32f *d_min, *d_max, *d_mean, *d_std;
        CUDA_CALL(cudaMalloc((void **) &d_min, sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &d_max, sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &d_mean, sizeof(float)));
        CUDA_CALL(cudaMalloc((void **) &d_std, sizeof(float)));

        // minmax
        NPP_CALL(nppsMinMax_32f(d_u, (int) size, d_min, d_max, d_buf1));
        CUDA_CALL(cudaMemcpy(h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

        // meanstd
        NPP_CALL(nppsMeanStdDev_32f(d_u, (int) size, d_mean, d_std, d_buf2));
        CUDA_CALL(cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_std, d_std, sizeof(float), cudaMemcpyDeviceToHost));

        // Free
        CUDA_CALL(cudaFree(d_min));
        CUDA_CALL(cudaFree(d_max));
        CUDA_CALL(cudaFree(d_mean));
        CUDA_CALL(cudaFree(d_std));
    }

    float interpol_linear
            (const float *image,
             float x,
             float y,
             float z,
             int nx,
             int ny)

    /*
     Linear 3D Interpolation, linear array
     */
    {
        long ipx, ipy, ipz;        /* coordinates rounded to lower integer */
        float vx1, vx2;            /* distances from real to rounded x-coordinate, value: 0 - 1 */
        float vy1, vy2;            /* same with y */
        float vz1, vz2;            /* same with z */
        float ip1, ip2, ip3, ip4;    /* auxiliary variables */

        ipx = (long) x;
        vx2 = x - (float) ipx;
        vx1 = 1 - vx2;
        ipy = (long) y;
        vy2 = y - (float) ipy;
        vy1 = 1 - vy2;
        ipz = (long) z;
        vz2 = z - (float) ipz;
        vz1 = 1 - vz2;

        ip1 = access_3d(image, ipx, ipy, ipz, nx, ny)
              + (access_3d(image, ipx + 1, ipy, ipz, nx, ny) - access_3d(image, ipx, ipy, ipz, nx, ny)) * vx2;
        ip2 = access_3d(image, ipx, ipy + 1, ipz, nx, ny) * vx1 + access_3d(image, ipx + 1, ipy + 1, ipz, nx, ny) * vx2;
        ip3 = access_3d(image, ipx, ipy, ipz + 1, nx, ny) * vx1 + access_3d(image, ipx + 1, ipy, ipz, nx, ny) * vx2;
        ip4 = access_3d(image, ipx, ipy + 1, ipz + 1, nx, ny) * vx1 +
              access_3d(image, ipx + 1, ipy + 1, ipz + 1, nx, ny) * vx2;

        return ((ip1 * vy1 + ip2 * vy2) * vz1 + (ip3 * vy1 + ip4 * vy2) * vz2);

    }/* interpol */

    /*--------------------------------------------------------------------------*/

    long roundAF(float value) {
        long floor;
        float diff;

        floor = (long) value;
        diff = value - (float) floor;

        if (diff >= 0.5) return floor + 1;
        if (diff <= -0.5) return floor - 1;
        return floor;
    }/* roundAF */
}
