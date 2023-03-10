#include "combi.h"

/*
Segmentation with Level sets 3D
Achilleas Frangakis
*/

/* ---------------------------------------------------------------------- */
void segm_combi
        (int     nx,        /* image dimension in x direction */
         int     ny,        /* image dimension in y direction */
         int     nz,        /* image dimension in z direction */
         float   alpha,     /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
         float   beta,      /* Amount of mc motion 0 < beta < 1 */
         float    ***u)     /* input: original image ;  output: smoothed */

/*
 Segmentation with Level sets
*/
{
    int    i, j, k;                 /* loop variables */
    float   ***f;                    /* u at old time level */
    float   grad_sqr;                /* |grad(f)|^2, time saver */
    float   curv;                    /* Curvature */
    float   lvset;                   /* Normal */
    float   f0_x, f0_y, f0_z, f0_xx, f0_xy, f0_yy, f0_xz, f0_yz, f0_zz;   /* derivatives */
    float   fm_x, fp_x, fm_y, fp_y, fm_z, fp_z;
    float   max_fm_x, min_fp_x, max_fm_y, min_fp_y,  max_fm_z, min_fp_z;
/* ---- allocate storage f ---- */

    f=f3tensor(0, nx+1, 0, ny+1,0, nz+1);

/* ---- copy u into f ---- */
    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
                f[i][j][k] = u[i][j][k];

/* ---- create periodic dummy boundaries for f ---- */
    dummies (f, nx, ny, nz, 1);

/* loop */
    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
            {
                /* central spatial derivatives */
                f0_x  = (f[i+1][j][k] - f[i-1][j][k]) / 2;
                f0_y  = (f[i][j+1][k] - f[i][j-1][k]) / 2;
                f0_z  = (f[i][j][k+1] - f[i][j][k-1]) / 2;
                f0_xx = (f[i+1][j][k] - 2.0f * f[i][j][k] + f[i-1][j][k]);
                f0_yy = (f[i][j+1][k] - 2.0f * f[i][j][k] + f[i][j-1][k]);
                f0_zz = (f[i][j][k+1] - 2.0f * f[i][j][k] + f[i][j][k-1]);

                if (f0_x * f0_y < 0.0f)
                    f0_xy = (   f[i+1][j+1][k] - f[i][j+1][k] - f[i+1][j][k] + f[i][j][k]
                                + f[i-1][j-1][k] - f[i][j-1][k] - f[i-1][j][k] + f[i][j][k] ) / 2;
                else
                    f0_xy = ( - f[i-1][j+1][k] + f[i][j+1][k] + f[i+1][j][k] - f[i][j][k]
                              - f[i+1][j-1][k] + f[i][j-1][k] + f[i-1][j][k] - f[i][j][k] ) / 2;

                if (f0_y * f0_z < 0.0f)
                    f0_yz = (   f[i][j+1][k+1] - f[i][j+1][k] - f[i][j][k+1] + f[i][j][k]
                                + f[i][j-1][k-1] - f[i][j-1][k] - f[i][j][k-1] + f[i][j][k] ) / 2;
                else
                    f0_yz = ( - f[i][j+1][k-1] + f[i][j+1][k] + f[i][j][k+1] - f[i][j][k]
                              - f[i][j-1][k+1] + f[i][j-1][k] + f[i][j][k-1] - f[i][j][k] ) / 2;

                if (f0_x * f0_z < 0.0f)
                    f0_xz = (   f[i+1][j][k+1] - f[i+1][j][k] - f[i][j][k+1] + f[i][j][k]
                                + f[i-1][j][k-1] - f[i-1][j][k] - f[i][j][k-1] + f[i][j][k] ) / 2;
                else
                    f0_xz = ( - f[i-1][j][k+1] + f[i][j][k+1] + f[i+1][j][k] - f[i][j][k]
                              - f[i+1][j][k-1] + f[i][j][k-1] + f[i-1][j][k] - f[i][j][k] ) / 2;

                grad_sqr = f0_x * f0_x + f0_y * f0_y + f0_z * f0_z;

                if (grad_sqr != 0.0f)
                    curv = 0.2f * (f0_x * f0_x * (f0_yy + f0_zz) + f0_y * f0_y * (f0_xx + f0_zz)
                                   + f0_z * f0_z * (f0_xx + f0_yy)  - 2.0f * f0_x * f0_y * f0_xy
                                   - 2.0f * f0_y * f0_z * f0_yz - 2.0f * f0_x * f0_z * f0_xz) / grad_sqr;
                else curv = 0.f;

                if (alpha >0)
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
                }
                else
                {
                    fm_x = f[i][j][k] - f[i-1][j][k];
                    fp_x = f[i+1][j][k] - f[i][j][k];
                    fm_y = f[i][j][k] - f[i][j-1][k];
                    fp_y = f[i][j+1][k] - f[i][j][k];
                    fm_z = f[i][j][k] - f[i][j][k-1];
                    fp_z = f[i][j][k+1] - f[i][j][k];

                    if (fm_x>0) max_fm_x = fm_x; else max_fm_x = 0;
                    if (fp_x<0) min_fp_x = fp_x; else min_fp_x = 0;
                    if (fm_y>0) max_fm_y = fm_y; else max_fm_y = 0;
                    if (fp_y<0) min_fp_y = fp_y; else min_fp_y = 0;
                    if (fm_z>0) max_fm_z = fm_z; else max_fm_z = 0;
                    if (fp_z<0) min_fp_z = fp_z; else min_fp_z = 0;

                    /* Level set in all directions */
                    lvset = 0.5f * sqrtf (max_fm_x * max_fm_x + min_fp_x * min_fp_x
                                          + max_fm_y * max_fm_y + min_fp_y * min_fp_y
                                          + max_fm_z * max_fm_z + min_fp_z * min_fp_z);

                }

                u[i][j][k] = f[i][j][k] +  beta * curv +  alpha * lvset;
            }

/* ---- disallocate storage for f ---- */
    free_f3tensor(f, 0, nx+1,0, ny+1,0, nz+1);

} /* segm */
/*--------------------------------------------------------------------------*/

void segm_combi_iterate(float ***u,                /* input/output image */
                        int pmax,                  /* iterations of mcm/levelset smoothing */
                        int nx,int ny, int nz,     /* image dimensions */
                        float alpha,               /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
                        float beta,                /* Amount of mc motion 0 < beta < 1 */
                        int verbose)               /* whether to compute and print image stats */
{
    float min, max, mean, vari;

    if(verbose) {
        analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
        printf("Input Data: min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", min, max, mean, vari);
    }

    for (int p=1; p<=pmax; p++)
    {
        /* perform one iteration */
        if(verbose) {
            printf("iteration number: %5d / %d \n", p, pmax);
        }
        segm_combi (nx, ny, nz, alpha, beta,  u);

        /* check minimum, maximum, mean, variance */
        if(verbose) {
            analyse (u, nx, ny, nz, &min, &max, &mean, &vari);
            printf("min: %1.6f, max: %1.6f, mean: %1.6f, variance: %1.6f\n", min, max, mean, vari);
        }
    }

}