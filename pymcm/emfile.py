import numpy as np
import io

EM_DATA_TYPE = {
    1: 'b',
    2: 'i2',
    4: 'i4',
    5: 'f4',
    8: 'c8',
    9: 'f8',
    10: 'f2'
}

EM_DATA_TYPE_inv = {np.dtype(v): k for k, v in EM_DATA_TYPE.items()}

EM_HEADER_DTYPE = np.dtype([
    ('MachineCoding', 'b'),      # Machine Coding
    ('NotUsed1', 'b'),      # Not used
    ('NotUsed2', 'b'),      # Not used
    ('DataType', 'b'),      # Data type coding

    ('DimX', 'i4'),      # Image size X
    ('DimY', 'i4'),     # Image size Y
    ('DimZ', 'i4'),  # Image size Z

    ('Comment', 'b', 80),  # Comment

    ('Voltage', 'i4'),  # accelerating Voltage (Factor 1000)
    ('Cs', 'i4'),  # Voltage (Factor 1000)
    ('Aperture', 'i4'),  # Aperture (Factor 1000)
    ('Magnification', 'i4'),  # End magnification (Factor 1)
    ('PostMagnification', 'i4'),  # Postmagnification of CCD (fixed value:1000!) (Factor 1000)
    ('ExposureTime', 'i4'),  # Exposure time in seconds (Factor 1000)
    ('ObjectPixelSize', 'i4'),  # Pixelsize in object-plane (Factor 1000)
    ('Microscope', 'i4'),  # EM-Code
    ('Pixelsize', 'i4'),  # Physical pixelsize on CCD (Factor 1000)
    ('CCDArea', 'i4'),  # Phys_pixel_size * nr_of_pixels (Factor 1000)
    ('Defocus', 'i4'),  # defocus, underfocus is neg. (Factor 1)
    ('Astigmatism', 'i4'),  # Astigmatism (Factor 1)
    ('AstigmatismAngle', 'i4'),  # Angle of astigmatism (Factor 1000)
    ('FocusIncrement', 'i4'),  # Focusincr. for focus-series (Factor 1)
    ('CountsPerElectron', 'i4'),  # Counts per primary electron, sensitivity of CCD (Factor 1000)
    ('Intensity', 'i4'),  # intensity value of C2 (Factor 1000)
    ('EnergySlitwidth', 'i4'),  # 0 for no slit, x>0 for positive slitwidth (Factor 1)
    ('EnergyOffset', 'i4'),  # Energy offset from zero-loss (Factor 1)
    ('Tiltangle', 'i4'),  # Tiltangle (Factor 1000)
    ('Tiltaxis', 'i4'),  # Axis perpend. to tiltaxis (Factor 1000)

    ('IsNewHeaderFormat', 'i4'),  # A value != 0 indicates that the newly added members have a meaning...
    ('AlignmentScore', 'f4'),  # Alignment error in pixels
    ('BeamDeclination', 'f4'),  # Beam declination (found by alignment or user provided) in deg
    ('MarkerOffset', 'i4'),  # Offest to marker positions (Appended to the actual EM file (32bit is enough)) (3*marker count float values)
    ('MagAnisotropyFactor', 'f4'),  # Magnification anisotropy amount
    ('MagAnisotropyAngle', 'f4'),  # Magnification anisotropy angle (in deg)
    ('ImageSizeX', 'i4'),  # Image dimension X
    ('ImageSizeY', 'i4'),  # Image dimension Y
    ('Fillup1', 'i4', 12),  # Fillup to 128 bytes
    ('Username', 'b', 20),  # Username (20 chars)
    ('Date', 'b', 8),  # Date (8 chars)
    ('Fillup2', 'b', 228),  # Fillup to 256 bytes
])

def emread(em_name):
    with open(em_name, "rb") as fin:
        # Read header
        header_array = bytearray(512)
        bytes_read = fin.readinto(header_array)
        header = np.frombuffer(header_array, dtype=EM_HEADER_DTYPE)

        # Check datatype
        datatype = header['DataType'][0]
        if datatype not in EM_DATA_TYPE.keys():
            raise ValueError("Unknown Data Type {} for file {}".format( datatype, em_name))

        npdtype = np.dtype(EM_DATA_TYPE[datatype])

        # Dimensions
        dimx = header['DimX'][0]
        dimy = header['DimY'][0]
        dimz = header['DimZ'][0]

        # Read data
        data_array = bytearray(dimx * dimy * dimz * npdtype.itemsize)
        bytes_read = fin.readinto(data_array)
        data = np.frombuffer(data_array, dtype=npdtype, count=dimx * dimy * dimz)
        data = data.reshape((dimz, dimy, dimx), order='C')
    return data

def emwrite(data, em_name):

    datatype = data.dtype
    if datatype not in EM_DATA_TYPE_inv.keys():
        raise ValueError("Can't write data of type {}".format(datatype))

    with open(em_name, "wb") as fout:
        if data.ndim == 1:
            dimx = data.shape[0]
            dimy = 1
            dimz = 1
        elif len(data.shape) == 2:
            dimx = data.shape[1]
            dimy = data.shape[0]
            dimz = 1
        else:
            dimx = data.shape[2]
            dimy = data.shape[1]
            dimz = data.shape[0]

        header = np.zeros((1), dtype=EM_HEADER_DTYPE)
        header['MachineCoding'][0] = 6
        header['DataType'][0] = EM_DATA_TYPE_inv[datatype]
        header['Pixelsize'][0] = 1
        header['DimX'][0] = dimx
        header['DimY'][0] = dimy
        header['DimZ'][0] = dimz

        fout.write(header.tobytes())
        fout.write(data.tobytes())


