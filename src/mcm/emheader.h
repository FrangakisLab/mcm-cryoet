#ifndef MCM_EMHEADER_H
#define MCM_EMHEADER_H

//! Machine Coding
enum EmMachine_Enum
{
    //! OS-9
    EMMACHINE_OS9 = 0,
    //! VAX
    EMMACHINE_VAX = 1,
    //! Convex
    EMMACHINE_CONVEX = 2,
    //! SGI
    EMMACHINE_SGI = 3,
    //! Mac
    EMMACHINE_MAC = 5,
    //! PC
    EMMACHINE_PC = 6
};

//! Microscope Coding
enum EmMicroscope_Enum
{
    //! Extern
    EMMICROSCOPE_EXTERN = 0,
    //! EM420
    EMMICROSCOPE_EM420 = 1,
    //! CM12
    EMMICROSCOPE_CM12 = 2,
    //! CM200
    EMMICROSCOPE_CM200 = 3,
    //! CM120/Biofilter
    EMMICROSCOPE_CM120BIOFILTER = 4,
    //! CM300
    EMMICROSCOPE_CM300 = 5,
    //! Polara
    EMMICROSCOPE_POLARA = 6
};

//! Data Type Coding
enum EmDataType_Enum
{
    //! Byte (1 byte)
    EMDATATYPE_BYTE = 1,
    //! Short (2 bytes)
    EMDATATYPE_SHORT = 2,
    //! Int (4 bytes)
    EMDATATYPE_INT = 4,
    //! Float (4 bytes)
    EMDATATYPE_FLOAT = 5,
    //! Complex (8 bytes)
    EMDATATYPE_COMPLEX = 8,
    //! Double (8 bytes)
    EMDATATYPE_DOUBLE = 9,
    //! Half float
    EMDATATYPE_HALF = 10
};


/*!
Structure of EM-Data Files:
-Byte 1: Machine Coding:       Machine:    Value:
OS-9         0
VAX          1
Convex       2
SGI          3
Mac          5
PC           6
-Byte 2: General purpose. On OS-9 system: 0 old version 1 is new version
-Byte 3: Not used in standard EM-format, if this byte is 1 the header is abandoned.
-Byte 4: Data Type Coding:         Image Type:     No. of Bytes:   Value:
byte            1               1
short           2               2
long int        4               4
float           4               5
complex         8               8
double          8               9
-Three long integers (3x4 bytes) are image size in x, y, z Dimension
-80 Characters as comment
-40 long integers (4 x 40 bytes) are user defined parameters
-256 Byte with userdata, first 20 chars username, 8 chars date (i.e.03/02/03)
-Raw data following with the x variable as the fastest dimension, then y and z
*/
//! Em file header
struct struct_EmHeader
{
    //! Machine Coding
    char  MachineCoding;
    //! Not used
    char NotUsed1;
    //! Not used
    char NotUsed2;
    //! Data type coding
    char DataType;

    //! Image size X
    int DimX;
    //! Image size Y
    int DimY;
    //! Image size Z
    int DimZ;

    //! Comment
    char Comment[80];

    //! accelerating Voltage (Factor 1000)
    int Voltage;
    //! Voltage (Factor 1000)
    int Cs;
    //! Aperture (Factor 1000)
    int Aperture;
    //! End magnification (Factor 1)
    int Magnification;
    //! Postmagnification of CCD (fixed value:1000!) (Factor 1000)
    int PostMagnification;
    //! Exposure time in seconds (Factor 1000)
    int ExposureTime;
    //! Pixelsize in object-plane (Factor 1000)
    int ObjectPixelSize;
    //! EM-Code
    int Microscope;
    //! Physical pixelsize on CCD (Factor 1000)
    int Pixelsize;
    //! Phys_pixel_size * nr_of_pixels (Factor 1000)
    int CCDArea;
    //! defocus, underfocus is neg. (Factor 1)
    int Defocus;
    //! Astigmatism (Factor 1)
    int Astigmatism;
    //! Angle of astigmatism (Factor 1000)
    int AstigmatismAngle;
    //! Focusincr. for focus-series (Factor 1)
    int FocusIncrement;
    //! Counts per primary electron, sensitivity of CCD (Factor 1000)
    int CountsPerElectron;
    //! intensity value of C2 (Factor 1000)
    int Intensity;
    //! 0 for no slit, x>0 for positive slitwidth (Factor 1)
    int EnergySlitwidth;
    //! Energy offset from zero-loss (Factor 1)
    int EnergyOffset;
    //! Tiltangle (Factor 1000)
    int Tiltangle;
    //! Axis perpend. to tiltaxis (Factor 1000)
    int Tiltaxis;
    //Added by MK for alignment info (08.03.2017):
    //! A value != 0 indicates that the newly added members have a meaning...
    int IsNewHeaderFormat;
    //! Alignment error in pixels
    float AlignmentScore;
    //! Beam declination (found by alignment or user provided) in deg
    float BeamDeclination;
    //! Offest to marker positions (Appended to the actual EM file (32bit is enough)) (3*marker count float values)
    int MarkerOffset;
    //! Magnification anisotropy amount
    float MagAnisotropyFactor;
    //! Magnification anisotropy angle (in deg)
    float MagAnisotropyAngle;
    //! Image dimension X
    int ImageSizeX;
    //! Image dimension Y
    int ImageSizeY;
    //! Fillup to 128 bytes
    int Fillup1[12];
    //! Username (20 chars)
    char Username[20];
    //! Date (8 chars)
    char Date[8];
    //! Fillup to 256 bytes
    char Fillup2[256 - 28];
};
typedef struct struct_EmHeader EmHeader;

#endif //MCM_EMHEADER_H
