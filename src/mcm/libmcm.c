#include "../../include/libmcm.h"
#include "combi.h"
#include "mcm.h"
#include "geodesic.h"

void run_combi_cpu(const float* input,       /* input image */
                   float* output,            /* output smoothed image */
                   int iterations,           /* iterations of mcm/levelset smoothing */
                   float alpha,              /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
                   float beta,               /* Amount of mc motion 0 < beta < 1 */
                   int nx, int ny, int nz,   /* image dimensions */
                   int verbose)              /* whether to compute and print image stats */
{
    float  ***u;                     /* image */
    int    i, j, k, c;               /* loop variable */
    int    pmax = iterations;        /* largest iteration number */

/* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* read image data */
    c = 0;
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                u[i][j][k] = input[c];
                c++;
            }

/* ---- process image ---- */
    segm_combi_iterate(u, pmax, nx, ny, nz, alpha, beta, verbose);

/* write image data */
    c = 0;
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++){
                output[c] = u[i][j][k];
                c++;
            }

/* ---- disallocate storage ---- */
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
}

void run_mcm_cpu(const float* input,             /* input image */
                 float* output,                  /* output smoothed image */
                 int iterations,                 /* iterations of mcm smoothing */
                 int nx, int ny, int nz,         /* image dimensions */
                 float hx, float hy, float hz,   /* pixel width in x,y,z direction */
                 int verbose)                    /* whether to compute and print image stats */
{
    float  ***u;                     /* image */
    int    i, j, k, c;               /* loop variable */
    int    pmax = iterations;        /* largest iteration number */

/* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* read image data */
    c = 0;

    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                u[i][j][k] = input[c];
                c++;
            }

/* ---- process image ---- */
    float ht = 0.2f;
    mcm_iterate(u, pmax, ht, nx, ny, nz, hx,hy, hz, verbose);

/* write image data */
    c = 0;
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++) {
                output[c] = u[i][j][k];
                c++;
            }

/* ---- disallocate storage ---- */
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
}

void run_trace_cpu(const float* speed_input,  /* input speed image */
                   float* output_vol,         /* output path image */
                   float* output_trace,       /* output path coordinates, allocate 3 x maxstep+1 */
                   int* tracelength,          /* actual output path length */
                   int x1, int x2, int x3,    /* end coordinate (one-based) */
                   int y1, int y2, int y3,    /* start coordinate (one-based) */
                   int nx, int ny, int nz,    /* image dimensions */
                   int maxstep,               /* maximum path length (good default: 10000) */
                   int verbose)               /* whether to compute and print image stats */
{
    float  ***u;			          /* image */
    float  ***g;			          /* velocity */
    int   i, j, k, c;		              /* loop variable */
    float  ***grdx, ***grdy, ***grdz;
    float  ***trace;
    //int    tracelength;

    /* allocate storage */
    u=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    g=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdx=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdy=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    grdz=f3tensor(0, nx+1, 0, ny+1,0, nz+1);
    trace = f3tensor(0, 3, 0, maxstep+1, 0, 1);

/* Initiate levelset with zeros */
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                u[i][j][k] =  0;
            }

/*Set start position of level set to 1 */
    u[x1][x2][x3]=1;

/*Read speed data */
    c = 0;
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                g[i][j][k] = speed_input[c];
                c++;
            }

    if (g[x1][x2][x3]<0.9 || g[y1][y2][y3]<0.9 )
    {
        printf("Error: Pixel values not equal to 1.\n");
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", x1, x2, x3, g[x1][x2][x3]);
        printf("Value at x1=%d, x2=%d, x3=%d : %f\n", y1, y2, y3, g[y1][y2][y3]);
        printf("exiting program\n");
        exit(1);
    }

/* ---- process image ---- */
    segm_iterate_trace(g, u, grdx, grdy, grdz, trace, tracelength,
                       nx, ny, nz,
                       x1, x2, x3,
                       y1, y2, y3,
                       maxstep,
                       verbose);

/* write volume data */
    c = 0;
    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++) {
                output_vol[c] = u[i][j][k];
                c++;
            }

    c = 0;
    for (k=1; k<=1; k++)
        for (j=1; j<=tracelength[0]; j++)
            for (i=1; i<=3; i++) {
                output_trace[c] = trace[i][j][k];
                c++;
            }

/* ---- disallocate storage ---- */
    free_f3tensor(u, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(g, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdx, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdy, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(grdz, 0, nx+1,0, ny+1,0, nz+1);
    free_f3tensor(trace, 0, 3, 0, maxstep+1, 0, 1);
}
