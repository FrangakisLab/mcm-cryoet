#ifndef MCM_COMBI_H
#define MCM_COMBI_H

#include "util.h"

void segm_combi
        (int     nx,        /* image dimension in x direction */
         int     ny,        /* image dimension in y direction */
         int     nz,        /* image dimension in z direction */
         float   alpha,     /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
         float   beta,      /* Amount of mc motion 0 < beta < 1 */
         float    ***u);    /* input: original image ;  output: smoothed */

void segm_combi_iterate(float ***u,                /* input/output image */
                        int pmax,                  /* iterations of mcm/levelset smoothing */
                        int nx,int ny, int nz,     /* image dimensions */
                        float alpha,               /* Amount and direction of levelset motion -1 < alpha < 1, negative moves inward */
                        float beta,                /* Amount of mc motion 0 < beta < 1 */
                        int verbose);              /* whether to compute and print image stats */

#endif //MCM_COMBI_H

