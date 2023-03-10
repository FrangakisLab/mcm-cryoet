#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nrutil.h"
#include "util.h"
#include "emfile.h"
#include "combi.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <npps.h>
#include "util.cuh"
#include "combi.cuh"
#endif

int main (int argc, char **argv)
{
    char   filename_in[180];         /* for reading data */
    char   filename_out[180];        /* for reading data */
    float  ***u;                     /* image */
    int    nx, ny, nz;               /* image size in x, y, z direction */
    EmHeader header;                 /* Header of the em-file */
    int    pmax;                     /* largest iteration number */
    float  max, min;                 /* largest, smallest grey value */
    float  mean;                     /* average grey value */
    float  vari;                     /* variance */
    float  alpha;                    /* Balance */
    float  beta;

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


    if (argc<5)
    {
        printf("Error: incorrect number of input arguments\n");
        printf("usage:\n");
        printf("levelSet <inputfile> <outputfile> <Iterations> <alpha> <beta> \n");
        printf("Only mean curvature motion alpha=0 and beta=1");
        printf("Only Level set motion alpha=1 and beta=0");
        printf(" -1<=alpha<=1 and 0<=beta<=1");
        printf(" -1<=alpha<0 inwards movement => Erosion");
        printf(" 0<=alpha<1 outwards movement => Dilations");

        printf("exiting program\n");
        exit(1);
    }

    strcpy(filename_in,argv[1]);
    strcpy(filename_out,argv[2]);

    printf("input file = %s\n", filename_in);
    printf("output file = %s\n", filename_out);

    pmax = atof(argv[3]);
    alpha = atof(argv[4]);
    beta = atof(argv[5]);


/* ---- read input image (em format) ---- */
    header = emread_header(filename_in);
    nx = header.DimX;
    ny = header.DimY;
    nz = header.DimZ;

    printf("Dimensions are %d x %d x %d\n", nx, ny, nz);

    printf("\n");
    printf("alpha (LevelSet Amount): %1.6f, beta (Curvature Amount): %1.6f\n", alpha, beta);
    printf("\n");

#ifdef USE_CUDA
/* allocate storage */
    size_t size = nx * ny * nz;
    size_t size_bytes = size * sizeof(float);
    h_u = malloc(size_bytes);
    CUDA_CALL(cudaMalloc((void**)&d_u, size_bytes));

/* read image data */
    emread_linear(filename_in, h_u);

    // Send to GPU
    CUDA_CALL(cudaMemcpy(d_u, h_u, size_bytes, cudaMemcpyHostToDevice));

    // Min/Max setup
    nppsMinMaxGetBufferSize_32f((int)size, &minmax_buf_size);
    CUDA_CALL(cudaMalloc((void**)&d_minmax, minmax_buf_size));

    // Mean/Std setup
    nppsMeanStdDevGetBufferSize_32f((int)size, &meastd_buf_size);
    CUDA_CALL(cudaMalloc((void**)&d_meastd, meastd_buf_size));

    analyse_CUDA(d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);

/* ---- Image ---- */
    printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", h_min, h_max, h_mean, h_std * h_std);
#else
/* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* read image data */
    emread_tensor(filename_in, u);

/* ---- Image ---- */
    analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
    printf("min: %1.6f, max: %1.6f, mean: %1.6f\n", min, max, mean);

#endif

/* ---- process image ---- */
int verbose = 1;
#ifdef USE_CUDA
    segm_combi_iterate_CUDA(d_u, pmax, nx, ny, nz, alpha, beta, d_minmax, d_meastd, verbose);
#else
    segm_combi_iterate(u, pmax, nx, ny, nz, alpha, beta, verbose);
#endif

/* ---- write output image (em format) ---- */
#ifdef USE_CUDA
    CUDA_CALL(cudaMemcpy(h_u, d_u, size_bytes, cudaMemcpyDeviceToHost));
    emwrite_linear(filename_out, h_u, &header, nx, ny, nz);
#else
/* write image data and close file */
    emwrite_tensor(filename_out, u, &header, nx, ny, nz);
#endif
    printf("output image %s successfully written\n\n", filename_out);
    printf("program finished\n");

/* ---- disallocate storage ---- */
#ifdef USE_CUDA
    free(h_u);

    CUDA_CALL(cudaFree(d_u));
    CUDA_CALL(cudaFree(d_minmax));
    CUDA_CALL(cudaFree(d_meastd));
#else
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
#endif

    return 1;
}

