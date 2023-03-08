#ifndef MCM_UTIL_H
#define MCM_UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nrutil.h"

void analyse
        (float   ***u,         /* image, unchanged */
         long    nx,          /* pixel number in x direction */
         long    ny,          /* pixel number in x direction */
         long    nz,          /* pixel number in x direction */
         float   *min,        /* minimum, output */
         float   *max,        /* maximum, output */
         float   *mean,       /* mean, output */
         float   *vari);       /* variance, output */

void dummies
        (float ***v,        /* image matrix */
         long  nx,          /* size in x direction */
         long  ny,          /* size in y direction */
         long  nz,           /* size in z direction */
         int   type);        /* 1 - periodic boundaries; 2 - extended boundaries */

float interpol
        (float   ***image,
         float   x,
         float   y,
         float   z);

long roundAF (float value);

#endif //MCM_UTIL_H
