#ifndef MCM_MCM_H
#define MCM_MCM_H

#include "util.h"

void mcm
        (float    ht,        /* time step size, 0 < ht <= 0.25 */
         int      nx,        /* image dimension in x direction */
         int      ny,        /* image dimension in y direction */
         int      nz,        /* image dimension in z direction */
         float    hx,        /* pixel width in x direction */
         float    hy,        /* pixel width in y direction */
         float    hz,        /* pixel width in y direction */
         float    ***u);       /* input: original image ;  output: smoothed */

void mcm_iterate(float ***u,
                 long pmax,
                 float ht,
                 int nx, int ny, int nz,
                 float hx, float hy, float hz,
                 int verbose);

#endif //MCM_MCM_H
