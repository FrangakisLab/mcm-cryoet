#include "emfile.h"

EmHeader emread_header(const char *filename)
{
    FILE  *infile;
    EmHeader header;

    if ((infile = fopen (filename, "r")) == 0)
    {
        printf("Error: could not open input file %s\n", filename);
        printf("exiting program\n");
        exit (1);
    }

/* read header */
    fread(&header, sizeof(EmHeader), 1, infile);

    return header;
}

/* ---------------------------------------------------------------------- */
void emread_tensor(const char *filename,
                   float ***data)
{
    FILE  *infile;
    float  hilf;			/* data points */
    int   i, j, k;		/* loop variable */
    int   nx, ny, nz;
    EmHeader header;

    if ((infile = fopen (filename, "r")) == 0)
    {
        printf("Error: could not open input file %s\n", filename);
        printf("exiting program\n");
        exit (1);
    }

/* read header */
    fread(&header, sizeof(EmHeader), 1, infile);

/* check for float format */
    if (header.DataType != EMDATATYPE_FLOAT)
    {
        printf ("Error: no float format given in data\n");
        printf("exiting program\n");
        exit(1);
    }
/* get dimensions of stack */
    nx = header.DimX;
    ny = header.DimY;
    nz = header.DimZ;

/* set file pointer to begin of data block */
    fseek (infile,512L,0);

    for (k=1; k<=nz; k++)
        for (j=1; j<=ny; j++)
            for (i=1; i<=nx; i++)
            {
                fread (&hilf, sizeof(float), 1, infile);
                data[i][j][k] =  hilf;
            }
    fclose(infile);
} /* emread_tensor */

/* ---------------------------------------------------------------------- */
void emwrite_tensor(const char *filename,
                    float ***outdata,
                    EmHeader *header,
                    int nx,
                    int ny,
                    int nz)
{
    FILE  *outfile;
    int   i, j, k;		/* loop variable */
    int noheader = 0;

    if (!header){
        noheader = 1;
        header = (EmHeader *) malloc(sizeof(EmHeader));
        memset(header, 0, sizeof(EmHeader));

        header->MachineCoding = EMMACHINE_PC;
        header->DataType = EMDATATYPE_FLOAT;
        header->Pixelsize = 1;
    }

    header->DimX = nx;
    header->DimY = ny;
    header->DimZ = nz;

    /* open output file, check if it is writable */
    if ((outfile = fopen (filename, "wb+")) == 0)
    {
        printf("Error: could not open output file %s\n", filename);
        printf("exiting program\n");
        exit (1);
    }

    fwrite(header, sizeof(EmHeader), 1, outfile);

/* write image data and close file */
    for (k = 1; k <= nz; k++)
        for (j = 1; j <= ny; j++)
            for (i = 1; i <= nx; i++)
                fwrite(&outdata[i][j][k], sizeof(float), 1, outfile);

    fclose(outfile);

    if(noheader==1){
        free(header);
    }
} /* emwrite_tensor */

/* ---------------------------------------------------------------------- */
void emread_linear
        (const char *filename,
         float* data)
{
    FILE  *infile;
    EmHeader header;
    int nx, ny, nz;
    //float *u;

    if ((infile = fopen (filename, "r")) == 0)
    {
        printf("Error: could not open input file %s\n", filename);
        printf("exiting program\n");
        exit (1);
    }

/* read header */
    fread (&header, sizeof(EmHeader), 1, infile);

/* check for float format */
    if (header.DataType != EMDATATYPE_FLOAT)
    {
        printf ("Error: no float format given in data\n");
        printf("exiting program\n");
        exit(1);
    }

/* get dimensions of stack */
    nx = header.DimX;
    ny = header.DimY;
    nz = header.DimZ;

/* set file pointer to begin of data block */
    fseek (infile,512L,0);

    fread (data, sizeof(float), nx * ny * nz, infile);
    fclose(infile);
} /* emread_linear */

/* ---------------------------------------------------------------------- */
void emwrite_linear
        (const char *filename,
         float *outdata,
         EmHeader *header,
         int nx,
         int ny,
         int nz)
{
    FILE  *outfile;
    int noheader = 0;

    if (!header){
        noheader = 1;
        header = (EmHeader *) malloc(sizeof(EmHeader));
        memset(header, 0, sizeof(EmHeader));

        header->MachineCoding = EMMACHINE_PC;
        header->DataType = EMDATATYPE_FLOAT;
        header->Pixelsize = 1;
    }

    header->DimX = nx;
    header->DimY = ny;
    header->DimZ = nz;

    /* open output file, check if it is writable */
    if ((outfile = fopen (filename, "wb+")) == 0)
    {
        printf("Error: could not open output file %s\n", filename);
        printf("exiting program\n");
        exit (1);
    }

    fwrite (header, sizeof(EmHeader), 1, outfile);

/* write image data and close file */
    fwrite (outdata, sizeof(float), nx * ny * nz, outfile);
    fclose(outfile);

    if (noheader==1){
        free(header);
    }
} /* emwrite_linear */