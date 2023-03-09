#include "mcm.h"

/*
Mean curvature Motion in 3D
Achilleas Frangakis
*/

/* ---------------------------------------------------------------------- */

void mcm
        (float    ht,        /* time step size, 0 < ht <= 0.25 */
         int     nx,        /* image dimension in x direction */
         int     ny,        /* image dimension in y direction */
         int     nz,        /* image dimension in z direction */
         float    hx,        /* pixel width in x direction */
         float    hy,        /* pixel width in y direction */
         float    hz,        /* pixel width in y direction */
         float    ***u)       /* input: original image ;  output: smoothed */

/*
 Mean curvature motion.
*/

{
    int    i, j, k;                    /* loop variables */
    float   fx, fy, fz, fxx, fxy, fyy, fxz, fyz, fzz;   /* derivatives */
    float   ***f;                     /* u at old time level */
    float   two_hx;                  /* 2.0 * hx, time saver */
    float   two_hy;                  /* 2.0 * hx, time saver */
    float   two_hz;                  /* 2.0 * hz, time saver */
    float   hx_sqr;                  /* hx * hx, time saver */
    float   hy_sqr;                  /* hy * hy, time saver */
    float   hz_sqr;                  /* hz * hz, time saver */
    float   two_hx_hy;               /* 2.0 * hx * hy, time saver */
    float   two_hx_hz;               /* 2.0 * hx * hz, time saver */
    float   two_hy_hz;               /* 2.0 * hy * hz, time saver */
    float   grad_sqr;                /* |grad(f)|^2, time saver */


/* ---- allocate storage f ---- */

    f=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* ---- copy u into f ---- */

    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
                f[i][j][k] = u[i][j][k];


/* ---- create periodic dummy boundaries for f ---- */

    dummies (f, nx, ny, nz, 1);


/* ---- loop ---- */

/* calculate some time savers */
    two_hx = 2.0f * hx;
    two_hy = 2.0f * hy;
    two_hz = 2.0f * hz;
    hx_sqr = hx * hx;
    hy_sqr = hy * hy;
    hz_sqr = hz * hz;
    two_hx_hy = 2.0f * hx * hy;
    two_hx_hz = 2.0f * hx * hz;
    two_hy_hz = 2.0f * hy * hz;

/* loop */
    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
            {
                /* central spatial derivatives */
                fx  = (f[i+1][j][k] - f[i-1][j][k]) / two_hx;
                fy  = (f[i][j+1][k] - f[i][j-1][k]) / two_hy;
                fz  = (f[i][j][k+1] - f[i][j][k-1]) / two_hz;
                fxx = (f[i+1][j][k] - 2.0f * f[i][j][k] + f[i-1][j][k]) / hx_sqr;
                fyy = (f[i][j+1][k] - 2.0f * f[i][j][k] + f[i][j-1][k]) / hy_sqr;
                fzz = (f[i][j][k+1] - 2.0f * f[i][j][k] + f[i][j][k-1]) / hz_sqr;

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



                grad_sqr = (fx * fx + fy * fy + fz * fz);
                if (grad_sqr != 0.0f)
                    u[i][j][k] = f[i][j][k] + ht *
                                              (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) + fz * fz * (fxx + fyy)
                                               - 2.0f * fx * fy * fxy - 2.0f * fy * fz * fyz - 2.0f * fx * fz * fxz) / grad_sqr;

            }


/* ---- disallocate storage for f ---- */

    free_f3tensor(f, 0, nx+1,0, ny+1,0, nz+1);

} /* mcm */
/*--------------------------------------------------------------------------*/

void mcm_iterate(float ***u,
                 long pmax,
                 float ht,
                 int nx, int ny, int nz,
                 float hx, float hy, float hz,
                 int verbose)
{
    float min, max, mean, vari;

    if(verbose) {
        analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
        printf("Input Data: min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", min, max, mean, vari);
    }

    for (long p=1; p<=pmax; p++)
    {
        /* perform one iteration */
        if(verbose) {
            printf("iteration number: %5ld / %ld \n", p, pmax);
        }

        mcm (ht, nx, ny, nz, hx, hy, hz, u);

        /* check minimum, maximum, mean, variance */
        if(verbose) {
            analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
            printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", min, max, mean, vari);
        }
    }
}