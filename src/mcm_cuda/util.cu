#include "util.cuh"

extern "C"
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
         float *h_std)
{
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

extern "C"
float interpol_linear
        (const float   *image,
         float   x,
         float   y,
         float   z,
         int     nx,
         int     ny)

/*
 Linear 3D Interpolation, linear array
 */
{
    long ipx, ipy, ipz;		/* coordinates rounded to lower integer */
    float vx1, vx2;			/* distances from real to rounded x-coordinate, value: 0 - 1 */
    float vy1, vy2;			/* same with y */
    float vz1, vz2;			/* same with z */
    float ip1, ip2, ip3, ip4;	/* auxiliary variables */

    ipx = (long)x;
    vx2 = x - (float)ipx;
    vx1 = 1 - vx2;
    ipy = (long)y;
    vy2 = y - (float)ipy;
    vy1 = 1 - vy2;
    ipz = (long)z;
    vz2 = z - (float)ipz;
    vz1 = 1 - vz2;

    ip1 = access_3d(image, ipx, ipy, ipz, nx, ny)
            + (access_3d(image, ipx+1, ipy, ipz, nx, ny) - access_3d(image, ipx, ipy, ipz, nx, ny)) * vx2;
    ip2 = access_3d(image, ipx, ipy+1, ipz, nx, ny) * vx1 + access_3d(image, ipx+1, ipy+1, ipz, nx, ny) * vx2;
    ip3 = access_3d(image, ipx, ipy, ipz+1, nx, ny) * vx1 + access_3d(image, ipx+1, ipy, ipz, nx, ny) * vx2;
    ip4 = access_3d(image, ipx, ipy+1, ipz+1, nx, ny) * vx1 + access_3d(image, ipx+1, ipy+1, ipz+1, nx, ny) * vx2;

    return ((ip1 * vy1 + ip2 * vy2) * vz1 + (ip3 * vy1 + ip4 * vy2) * vz2);

}/* interpol */

/*--------------------------------------------------------------------------*/

extern "C"
long roundAF (float value) {
    long floor;
    float diff;

    floor = (long) value;
    diff = value - (float) floor;

    if (diff >= 0.5) return floor+1;
    if (diff <= -0.5) return floor-1;
    return floor;
}/* roundAF */

