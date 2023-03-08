#include "util.h"

/* ---------------------------------------------------------------------- */
void analyse
        (float   ***u,         /* image, unchanged */
         long    nx,          /* pixel number in x direction */
         long    ny,          /* pixel number in x direction */
         long    nz,          /* pixel number in x direction */
         float   *min,        /* minimum, output */
         float   *max,        /* maximum, output */
         float   *mean,       /* mean, output */
         float   *vari)       /* variance, output */

/*
 calculates minimum, maximum, mean and variance of an image u
*/

{
    long    i, j, k;       /* loop variables */
    float   help;       /* auxiliary variable */
    float  help2;      /* auxiliary variable */

    *min  = u[1][1][1];
    *max  = u[1][1][1];
    help2 = 0.0f;

    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
            {
                if (u[i][j][k] < *min) *min = u[i][j][k];
                if (u[i][j][k] > *max) *max = u[i][j][k];
                help2 = help2 + (float)u[i][j][k];
            }
    *mean = (float)help2 / (float)(nx * ny * nz);

    *vari = 0.0f;
    for (i=1; i<=nx; i++)
        for (j=1; j<=ny; j++)
            for (k=1; k<=nz; k++)
            {
                help  = u[i][j][k] - *mean;
                *vari = *vari + help * help;
            }
    *vari = *vari / (float)(nx * ny * nz);

} /* analyse */

/* ---------------------------------------------------------------------- */

void dummies

        (float ***v,        /* image matrix */
         long  nx,          /* size in x direction */
         long  ny,          /* size in y direction */
         long  nz,           /* size in z direction */
         int   type)        /* 1 - periodic boundaries; 2 - reflected boundaries */

/* creates dummy boundaries by periodical or reflected continuation */

{
    switch (type) {
        case 1:  // Periodic boundaries
        {
            long i, j, k;  /* loop variables */

            for (i = 1; i <= nx; i++) {
                for (j = 1; j <= ny; j++) {
                    v[i][j][0] = v[i][j][nz];    /* first level of the extended image is */
                    /* equal to the last level of the       */
                    /* original image                       */
                    v[i][j][nz + 1] = v[i][j][1];
                }
            }

            for (j = 1; j <= ny; j++) {
                for (k = 0; k <= nz + 1; k++) {
                    v[0][j][k] = v[nx][j][k];
                    v[nx + 1][j][k] = v[1][j][k];
                }
            }

            for (k = 0; k <= nz + 1; k++) {
                for (i = 0; i <= nx + 1; i++) {
                    v[i][0][k] = v[i][ny][k];
                    v[i][ny + 1][k] = v[i][1][k];
                }
            }
            break;
        }
        case 2: // Extended boundaries
        {
            long i, j, k;  /* loop variables */

            for (i=1; i<=nx; i++)
            {
                for (j=1; j<=ny; j++)
                {
                    v[i][j][0]    = v[i][j][1];    /* first level of the extended image is */
                    /* equal to the last level of the       */
                    /* original image                       */
                    v[i][j][nz+1] = v[i][j][nz];
                }
            }

            for (j=1; j<=ny; j++)
            {
                for (k=0; k<=nz+1; k++)
                {
                    v[0][j][k]    = v[1][j][k];
                    v[nx+1][j][k] = v[nx][j][k];
                }
            }

            for (k=0; k<=nz+1; k++)
            {
                for (i=0; i<=nx+1; i++)
                {
                    v[i][0][k]    = v[i][1][k];
                    v[i][ny+1][k] = v[i][ny][k];
                }
            }
            break;
        }
    } // end switch
} /* dummies */

/* ---------------------------------------------------------------------- */

float interpol
        (float   ***image,
         float   x,
         float   y,
         float   z)

/*
 Linear 3D Interpolation
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

    ip1 = image[ipx][ipy][ipz] + (image[ipx+1][ipy][ipz]-image[ipx][ipy][ipz])*vx2;
    ip2 = image[ipx][ipy+1][ipz]*vx1 + image[ipx+1][ipy+1][ipz]*vx2;
    ip3 = image[ipx][ipy][ipz+1]*vx1 + image[ipx+1][ipy][ipz+1]*vx2;
    ip4 = image[ipx][ipy+1][ipz+1]*vx1 + image[ipx+1][ipy+1][ipz+1]*vx2;

    return ((ip1 * vy1 + ip2 * vy2) * vz1 + (ip3 * vy1 + ip4 * vy2) * vz2);

}/* interpol */

/*--------------------------------------------------------------------------*/

long roundAF (float value) {
    long floor;
    float diff;

    floor = (long) value;
    diff = value - (float) floor;

    if (diff >= 0.5) return floor+1;
    if (diff <= -0.5) return floor-1;
    return floor;
}/* roundAF */

