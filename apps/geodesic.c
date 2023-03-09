#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nrutil.h"
#include "util.h"
#include "emfile.h"
#include "geodesic.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <npps.h>
#include "util.cuh"
#include "geodesic.cuh"
#endif

/*
Path tracing with Level sets 3D
Achilleas Frangakis

Last change: 10/02/03 M. Riedlberger
Linear Interpolation of gradients,
assigning gray values to 8 discrete neighbor pixels
 *
 *Last edited 23-9-2021
*/

/*--------------------------------------------------------------------------*/
int main (int argc, char **argv)
{
    char   filename_out[180];	      /* for reading data */
    char   filename_speed_in[180];    /* for reading data */
    float  ***u;			          /* image */
    float  ***g;			          /* velocity */
    int   i, j, k;		              /* loop variable */
    int   nx, ny, nz;		          /* image size in x, y, z direction */
    EmHeader  header;		          /* Header of the em-file */
    float  max, min;		          /* largest, smallest grey value */
    float  mean;			          /* average grey value */
    float  vari;			          /* variance */
    float  alpha;			          /* Balance */
    int   x1,x2,x3,y1,y2,y3;	      /* Input and output coordinates */
    float  ***grdx, ***grdy, ***grdz; /* Gradients */
    float  ***trace;                  /* Trace coordinates */
    int    tracelength;               /* Trace length */
    float  gc1, gc2, gc3;		      /* step width in x, y, z */
    float  y1f, y2f, y3f;		      /* exact path coordinates */
    long   ipx, ipy, ipz;		      /* coordinates rounded to lower integer */
    float  vx1, vx2;		          /* distances from real to rounded x-coordinate, value: 0 - 1 */
    float  vy1, vy2;		          /* same with y */
    float  vz1, vz2;		          /* same with z */

#ifdef USE_CUDA
    float *h_u;                       /* linear host storage speed*/
    float *d_u;                       /* linear device storage image*/

    float *h_g;                       /* linear host storage speed*/
    float *d_g;                       /* linear device storage speed*/

    float *h_trace;                   /* linear host storage trace*/

    float *d_grdx;                    /* linear device storage gradient x*/
    float *d_grdy;                    /* linear device storage gradient z*/
    float *d_grdz;                    /* linear device storage gradient z*/

    int minmax_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_minmax;                  /* device buffer */
    float h_min, h_max;               /* host output */

    int meastd_buf_size = 0;          /* scratch buffer size */
    Npp8u *d_meastd;                  /* device buffer */
    float h_mean, h_std;              /* host output */
#endif

    if (argc<9)
    {
        printf("Error: incorrect number of input arguments\n");
        printf("Usage: geodesicLevelSets <Input file 'speed'> <Output file 'path'> <x,y,z Start coordinates> <x,y,z End coordinates>\n");
        printf("exiting program\n");
        printf("\n");
        printf("\n");
        printf("/Users/frangak/MyData/Code_Development/C/Dilation/geodesicLevelSets apis6/mask.em $apis6/mask.path1.em 480 440 7 70 70 41 \n");
        printf("\n");

        exit(1);
    }

    strcpy(filename_speed_in, argv[1]);
    strcpy(filename_out, argv[2]);

    printf("input speed file = %s\n", filename_speed_in);
    printf("output file = %s\n", filename_out);

    x1 = atof(argv[3]);
    x2 = atof(argv[4]);
    x3 = atof(argv[5]);

    y1 = atof(argv[6]);
    y2 = atof(argv[7]);
    y3 = atof(argv[8]);

/*Read speed file */
    header = emread_header(filename_speed_in);
    nx = header.DimX;
    ny = header.DimY;
    nz = header.DimZ;
    printf("Dimensions are %d x %d x %d\n", nx, ny, nz);

/* ---- read other parameters ---- */
    int maxstep = 10000;

#ifdef USE_CUDA
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
    cudaMallocHost((void**)&h_g, size_bytes);
    cudaMalloc((void**)&d_g, size_bytes);

    cudaMallocHost((void**)&h_u, size_bytes);
    cudaMalloc((void**)&d_u, size_bytes);

    cudaMallocHost((void**)&h_trace, 3*maxstep*sizeof(float));

    cudaMalloc((void**)&d_grdx, size_bytes);
    cudaMalloc((void**)&d_grdy, size_bytes);
    cudaMalloc((void**)&d_grdz, size_bytes);

    cudaMemset(d_grdx, 0, size_bytes);
    cudaMemset(d_grdy, 0, size_bytes);
    cudaMemset(d_grdz, 0, size_bytes);

/*Read speed file */
    emread_linear(filename_speed_in, h_g);

/* Initiate levelset with zeros */
    cudaMemset(d_u, 0, size_bytes);
    float one = 1.f;

/*Set start position of level set to 1 */
    cudaMemcpy(&access_3d(d_u, x1, x2, x3, nx, ny), &one, sizeof(float), cudaMemcpyHostToDevice);

/*Send speed file to device */
    cudaMemcpy(d_g, h_g, size_bytes, cudaMemcpyHostToDevice);

    if (access_3d(h_g, x1, x2, x3, nx, ny)<0.9 || access_3d(h_g, y1, y2, y3, nx, ny)<0.9 )
    {
        printf("Error: Pixel values not equal to 1.\n");
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", x1+1, x2+1, x3+1, access_3d(h_g, x1+1, x2+1, x3+1, nx, ny));
        printf("Value at y1=%d, y2=%d, y3=%d : %f\n", y1+1, y2+1, y3+1, access_3d(h_g, y1+1, y2+1, y3+1, nx, ny));
        printf("exiting program\n");
        exit(1);
    }

/* ---- Image ---- */
    // Min/Max setup
    nppsMinMaxGetBufferSize_32f((int)size, &minmax_buf_size);
    cudaMalloc((void**)&d_minmax, minmax_buf_size);

    // Mean/Std setup
    nppsMeanStdDevGetBufferSize_32f((int)size, &meastd_buf_size);
    cudaMalloc((void**)&d_meastd, meastd_buf_size);

    analyse_CUDA (d_u, nx, ny, nz, d_minmax, d_meastd, &h_min, &h_max, &h_mean, &h_std);
    printf("min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n", h_min, h_max, h_mean, h_std*h_std);
#else

/* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    g=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdx=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdy=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdz=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    trace = f3tensor(0, 3, 0, maxstep+1, 0, 1);

/*Read speed file */
    emread_tensor(filename_speed_in, g);

/* Initiate levelset with zeros */
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                u[i][j][k] =  0;
            }

/*Set start position of level set to 1 */
    u[x1][x2][x3]=1;


    if (g[x1][x2][x3]<0.9 || g[y1][y2][y3]<0.9 )
    {
        printf("Error: Pixel values not equal to 1.\n");
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", x1, x2, x3, g[x1][x2][x3]);
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", y1, y2, y3, g[y1][y2][y3]);
        printf("exiting program\n");
        exit(1);
    }


/* ---- Image ---- */
    analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
    printf("min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n", min, max, mean, vari);
#endif

/* ---- process image ---- */
int verbose = 1;
#ifdef USE_CUDA
    segm_iterate_trace_cuda(d_g, d_u, h_u, d_grdx, d_grdy, d_grdz,
                            h_trace, &tracelength,
                            nx, ny, nz,
                            x1, x2, x3,
                            y1, y2, y3,
                            d_minmax,
                            d_meastd,
                            maxstep,
                            verbose);
#else
    segm_iterate_trace(g, u, grdx, grdy, grdz, trace, &tracelength,
                       nx, ny, nz,
                       x1, x2, x3,
                       y1, y2, y3,
                       maxstep,
                       verbose);
#endif


/* write image data and close file */
#ifdef USE_CUDA
    emwrite_linear(filename_out, h_u, &header, nx, ny, nz);

    char path_out[360];
    sprintf(path_out, "%s_path.em", filename_out);

    emwrite_linear(path_out, h_trace, NULL, 3, tracelength, 1);
#else
    emwrite_tensor(filename_out, u, &header, nx, ny, nz);

    char path_out[360];
    sprintf(path_out, "%s_path.em", filename_out);

    emwrite_tensor(path_out, trace, NULL, 3, tracelength, 1);
#endif
    printf("output image %s successfully written\n\n", filename_out);
    printf("program finished\n");

/* ---- disallocate storage ---- */
#ifdef USE_CUDA
    cudaFreeHost(h_g);
    cudaFreeHost(h_u);
    cudaFreeHost(h_trace);

    cudaFree(d_u);
    cudaFree(d_g);
    cudaFree(d_grdx);
    cudaFree(d_grdy);
    cudaFree(d_grdz);
    cudaFree(d_minmax);
    cudaFree(d_meastd);
#else
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(g, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdx, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdy, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdz, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(trace, 0, 3, 0, maxstep+1, 0, 1);
#endif
}

