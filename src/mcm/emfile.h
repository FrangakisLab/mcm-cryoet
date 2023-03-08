#ifndef MCM_EMFILE_H
#define MCM_EMFILE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "nrutil.h"
#include "emheader.h"

EmHeader emread_header(const char *filename);

void emread_tensor(const char *filename,
                       float ***data);

void emwrite_tensor(const char *filename,
                    float ***outdata,
                    EmHeader *header,
                    int nx,
                    int ny,
                    int nz);

void emread_linear
        (const char *filename,
         float* data);

void emwrite_linear
        (const char *filename,
         float *outdata,
         EmHeader *header,
         int nx,
         int ny,
         int nz);


#endif //MCM_EMFILE_H
