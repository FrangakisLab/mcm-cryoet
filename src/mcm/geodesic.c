#include "geodesic.h"

/*
Path tracing with Level sets 3D
Achilleas Frangakis

Last change: 10/02/03 M. Riedlberger
Linear Interpolation of gradients,
assigning gray values to 8 discrete neighbor pixels
 *
 *Last edited 23-9-2021
*/

/* ---------------------------------------------------------------------- */

void segm_geo
        (long     nx,        /* image dimension in x direction */
         long     ny,        /* image dimension in y direction */
         long     nz,        /* image dimension in z direction */
         float    alpha,     /* Balance*/
         float    ***g,      /* velocity */
         float    ***u,      /* input: original image ;  output: smoothed */
         float    ***grdx,
         float    ***grdy,
         float    ***grdz)

/*
 Segmentation with Level sets
*/

{
    long    i, j, k;                 /* loop variables */
    float   ***f;                    /* u at old time level */
    float   grad_sqr;                /* |grad(f)|^2, time saver */
    float   lvset;                   /* Normal */
    float   f0_x, f0_y, f0_z, f0_xx, f0_xy, f0_yy, f0_xz, f0_yz, f0_zz;   /* derivatives */
    float   fm_x, fp_x, fm_y, fp_y, fm_z, fp_z;
    float   max_fm_x, min_fp_x, max_fm_y, min_fp_y, max_fm_z, min_fp_z;
    /* u at old time level */
/* ---- allocate storage f ---- */

    f=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* ---- copy u into f ---- */

    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
                f[i][j][k] = u[i][j][k];


/* ---- create reflecting dummy boundaries for f ---- */

    dummies (f, nx, ny, nz, 2);


    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
            {

                fm_x = f[i][j][k] - f[i-1][j][k];
                fp_x = f[i+1][j][k] - f[i][j][k];
                fm_y = f[i][j][k] - f[i][j-1][k];
                fp_y = f[i][j+1][k] - f[i][j][k];
                fm_z = f[i][j][k] - f[i][j][k-1];
                fp_z = f[i][j][k+1] - f[i][j][k];

                if (fm_x<0) max_fm_x = fm_x; else max_fm_x = 0;
                if (fp_x>0) min_fp_x = fp_x; else min_fp_x = 0;
                if (fm_y<0) max_fm_y = fm_y; else max_fm_y = 0;
                if (fp_y>0) min_fp_y = fp_y; else min_fp_y = 0;
                if (fm_z<0) max_fm_z = fm_z; else max_fm_z = 0;
                if (fp_z>0) min_fp_z = fp_z; else min_fp_z = 0;

                /* Level set in all directions */
                lvset = 0.5f * sqrtf (max_fm_x * max_fm_x + min_fp_x * min_fp_x
                                    + max_fm_y * max_fm_y + min_fp_y * min_fp_y
                                    + max_fm_z * max_fm_z + min_fp_z * min_fp_z);


                u[i][j][k] = f[i][j][k] + g[i][j][k] * lvset;


                if (u[i][j][k]>0.2 && f[i][j][k]<0.2) {
                    f0_x  = (f[i+1][j][k] - f[i-1][j][k]) / 2;
                    f0_y  = (f[i][j+1][k] - f[i][j-1][k]) / 2;
                    f0_z  = (f[i][j][k+1] - f[i][j][k-1]) / 2;
                    grad_sqr = sqrtf(f0_x * f0_x + f0_y * f0_y + f0_z * f0_z);
                    if (grad_sqr != 0.0) {
                        grdx[i][j][k]=f0_x/grad_sqr;
                        grdy[i][j][k]=f0_y/grad_sqr;
                        grdz[i][j][k]=f0_z/grad_sqr;
                    }
                }



            }

/* ---- disallocate storage for f ---- */
    free_f3tensor(f, 0, nx+1,0, ny+1,0, nz+1);

} /* segm */
/*--------------------------------------------------------------------------*/

void segm_iterate_trace
        (float    ***g,      /* velocity */
         float    ***u,      /* input: original image ;  output: smoothed */
         float    ***grdx,
         float    ***grdy,
         float    ***grdz,
         float    ***trace,
         int      *tracelength,
         int nx, int ny, int nz,
         int x1, int x2, int x3,
         int y1, int y2, int y3,
         int maxstep,
         int verbose)
{
    int   i, j, k;		/* loop variable */
    float  max, min;		/* largest, smallest grey value */
    float  mean;			/* average grey value */
    float  vari;			/* variance */
    float  alpha;			/* Balance */
    float  gc1, gc2, gc3;		/* step width in x, y, z */
    float  y1f, y2f, y3f;		/* exact path coordinates */
    long   ipx, ipy, ipz;		/* coordinates rounded to lower integer */
    float  vx1, vx2;		/* distances from real to rounded x-coordinate, value: 0 - 1 */
    float  vy1, vy2;		/* same with y */
    float  vz1, vz2;		/* same with z */

    if (verbose) {
        analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
        printf("Input Data: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n\n", min, max, mean, vari);
    }

    int c = 0;
    while (u[y1][y2][y3]<0.2)
    {
        /* perform one iteration */
        segm_geo (nx, ny, nz, alpha, g, u, grdx, grdy, grdz);
        c++;

        /* check minimum, maximum, mean, variance */
        if (verbose) {
            analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
            printf("LevelSet iter %d: min: %3.6f, max: %3.6f, mean: %3.6f, variance: %3.6f\n", c, min, max, mean, vari);
        }
    } /* while */

    for (k=0; k<=nz+1; k++)
        for (j=0; j<=ny+1; j++)
            for (i=0; i<=nx+1; i++)
                u[i][j][k] =  0;

    u[y1][y2][y3]=1;
    y1f=(float)y1; y2f=(float)y2; y3f=(float)y3;

    if (verbose) {
        printf("%f, %f, %f  - %d %d %d\n", y1f, y2f, y3f, y1, y2, y3);
        fflush(stdout);
    }

    c = 0;
    trace[1][c+1][1] = y1f;
    trace[2][c+1][1] = y2f;
    trace[3][c+1][1] = y3f;

    while (x1!=roundAF(y1f) || x2!=roundAF(y2f) || x3!=roundAF(y3f))
    {
        /* calulate exact stepwidth in each direction for next step */
        gc1 = interpol (grdx, y1f, y2f, y3f);
        gc2 = interpol (grdy, y1f, y2f, y3f);
        gc3 = interpol (grdz, y1f, y2f, y3f);

        /* calulate exact coordinates of next step */
        y1f=y1f+gc1;
        y2f=y2f+gc2;
        y3f=y3f+gc3;

        /* calulate shares of neighbor positions in each direction */
        ipx = (long)y1f;
        vx2 = y1f - (float)ipx;
        vx1 = 1 - vx2;
        ipy = (long)y2f;
        vy2 = y2f - (float)ipy;
        vy1 = 1 - vy2;
        ipz = (long)y3f;
        vz2 = y3f - (float)ipz;
        vz1 = 1 - vz2;

        /* calulate grey values of 8 neighbor pixels  */
        u[ipx][ipy][ipz] += (vx1*vy1*vz1);
        u[ipx+1][ipy][ipz] += (vx2*vy1*vz1);
        u[ipx][ipy+1][ipz] += (vx1*vy2*vz1);
        u[ipx+1][ipy+1][ipz] += (vx2*vy2*vz1);
        u[ipx][ipy][ipz+1] += (vx1*vy1*vz2);
        u[ipx+1][ipy][ipz+1] += (vx2*vy1*vz2);
        u[ipx][ipy+1][ipz+1] += (vx1*vy2*vz2);
        u[ipx+1][ipy+1][ipz+1] += (vx2*vy2*vz2);

        c++;
        trace[1][c+1][1] = y1f;
        trace[2][c+1][1] = y2f;
        trace[3][c+1][1] = y3f;

        if(verbose) {
            printf("Trace Step %d: %f, %f, %f\n", c, y1f, y2f, y3f);
            fflush(stdout);
        }

        if (c == maxstep-1){
            break;
        }
    }

    if(verbose) {
        printf("Terminating after taking %d steps.\n", c);
    }

    tracelength[0] = c+1;
}
