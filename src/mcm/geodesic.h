#ifndef MCM_GEODESIC_H
#define MCM_GEODESIC_H

#include "util.h"

void segm_geo
        (long     nx,        /* image dimension in x direction */
         long     ny,        /* image dimension in y direction */
         long     nz,        /* image dimension in z direction */
         float    alpha,     /* Balance*/
         float    ***g,      /* velocity */
         float    ***u,      /* input: original image ;  output: smoothed */
         float    ***grdx,
         float    ***grdy,
         float    ***grdz);

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
         int verbose);

#endif //MCM_GEODESIC_H