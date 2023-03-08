#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nrutil.h"
#include "util.h"
#include "emfile.h"
#include "mcm.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <npps.h>
#include "util.cuh"
#include "mcm.cuh"
#endif

int main (int argc, char **argv)
{
    char   filename_in[180];     /* for reading data */
    char   filename_out[180];    /* for reading data */
    float  ***u;                 /* image */
    int   nx, ny, nz;            /* image size in x, y, z direction */
    EmHeader header;
    float  ht;                   /* time step size */
    long   pmax;                 /* largest iteration number */
    float  max, min;             /* largest, smallest grey value */
    float  mean;                 /* average grey value */
    float  vari;                 /* variance */

#ifdef USE_CUDA
    float *h_u;                       /* linear host storage */
    float *d_u;                       /* linear device storage */

    int minmax_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_minmax;                  /* device buffer */
    float h_min, h_max;               /* host output */

    int meastd_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_meastd;                  /* device buffer */
    float h_mean, h_std;              /* host output */
#endif

    if (argc<4)
    {
        printf("Error: incorrect number of input arguments\n");
        printf("usage:\n");
        printf("mcm_3d <inputfile> <outputfile> <iterations>\n");
        printf("exiting program\n");
        exit(1);
    }

    strcpy(filename_in,argv[1]);
    strcpy(filename_out,argv[2]);

    printf("input file = %s\n", filename_in);
    printf("output file = %s\n", filename_out);

    pmax = atol(argv[3]); /* number of iterations */

/* ---- read other parameters ---- */
    ht=0.2f;

/* ---- read input image (em format) ---- */
    header = emread_header(filename_in);
    nx = header.DimX;
    ny = header.DimY;
    nz = header.DimZ;
    printf("Dimensions are %d x %d x %d\n", nx, ny, nz);

#ifdef USE_CUDA
/* allocate storage */
    size_t size = nx * ny * nz * sizeof(float);
    size_t size_bytes = size * sizeof(float);
    h_u = malloc(size_bytes);
    cudaMalloc((void**)&d_u, size_bytes);



/* read image data */
    emread_linear(filename_in, h_u);

    // Send to GPU
    cudaMemcpy(d_u, h_u, size_bytes, cudaMemcpyHostToDevice);

    // Min/Max setup
    nppsMinMaxGetBufferSize_32f((int)size, &minmax_buf_size);
    cudaMalloc((void**)&d_minmax, minmax_buf_size);

    // Mean/Std setup
    nppsMeanStdDevGetBufferSize_32f((int)size, &meastd_buf_size);
    cudaMalloc((void**)&d_meastd, meastd_buf_size);

    analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
    printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", h_min, h_max, h_mean, h_std*h_std);

/* ---- Image ---- */
    printf("minimum:       %3.6f \n", h_min);
    printf("maximum:       %3.6f \n", h_max);
    printf("mean:          %3.6f \n", h_mean);
    printf("variance:      %3.6f \n\n", h_std*h_std);
#else
/* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* read image data */
    emread_tensor(filename_in, u);

/* ---- Image ---- */
    analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
    printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", min, max, mean, vari);
#endif

/* ---- process image ---- */
int verbose = 1;
#ifdef USE_CUDA
    mcm_iterate_CUDA(d_u, pmax, ht, nx, ny, nz, 1.f, 1.f, 1.f, d_minmax, d_meastd, verbose);
#else
    mcm_iterate(u, pmax, ht, nx, ny, nz, 1.f, 1.f, 1.f, verbose);
#endif


/* write image data and close file */
#ifdef USE_CUDA
    cudaMemcpy(h_u, d_u, size_bytes, cudaMemcpyDeviceToHost);
    emwrite_linear(filename_out, h_u, &header, nx, ny, nz);
#else
    emwrite_tensor(filename_out, u, &header, nx, ny, nz);
#endif
    printf("output image %s successfully written\n\n", filename_out);
    printf("program finished\n");

/* ---- disallocate storage ---- */
#ifdef USE_CUDA
    free(h_u);

    cudaFree(d_u);
    cudaFree(d_minmax);
    cudaFree(d_meastd);
#else
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
#endif

    return 1;
}
