import numpy as np

cimport numpy as np
from libc.stdio cimport printf

np.import_array()

ctypedef np.float32_t float32_t

cdef extern from "libmcm_cuda.cuh":
    void run_combi_gpu(const float* h_input,       # input image
                       float* output,              # output smoothed image
                       int iterations,             # iterations of mcm/levelset smoothing
                       float alpha,                # Amount and direction of levelset motion -1 < alpha < 1, negative moves inward
                       float beta,                 # Amount of mc motion 0 < beta < 1
                       int nx, int ny, int nz,     # image dimensions
                       int verbose);               # whether to compute and print image stats

    void run_mcm_gpu(const float* input,              # input image
                     float* output,                   # output smoothed image
                     int iterations,                  # iterations of mcm smoothing
                     int nx, int ny, int nz,          # image dimensions
                     float hx, float hy, float hz,    # pixel width in x,y,z direction
                     int verbose);                    # whether to compute and print image stats

    void run_trace_gpu(const float* speed_input,  # input speed image
                       float* output_vol,         # output path image
                       float* output_trace,       # output path coordinates, allocate 3 x maxstep+1
                       int* tracelength,          # actual output path length
                       int x1, int x2, int x3,    # end coordinate (one-based)
                       int y1, int y2, int y3,    # start coordinate (one-based)
                       int nx, int ny, int nz,    # image dimensions
                       int maxstep,               # maximum path length (good default: 10000)
                       int verbose);              # whether to compute and print image stats

def combi_gpu(np.ndarray[float32_t, ndim=3] input,
              np.ndarray[float32_t, ndim=3] output,
              int iterations,
              float alpha, float beta,
              int verbose):

    cdef int nxi = input.shape[0]
    cdef int nyi = input.shape[1]
    cdef int nzi = input.shape[2]

    cdef int nxo = output.shape[0]
    cdef int nyo = output.shape[1]
    cdef int nzo = output.shape[2]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> input.data
    cdef float *pout = <float *> output.data

    run_combi_gpu(pinp, pout, iterations, alpha, beta, nxi, nyi, nzi, verbose)

def mcm_gpu(np.ndarray[float32_t, ndim=3] input,
            np.ndarray[float32_t, ndim=3] output,
            int iterations,
            float hx, float hy, float hz,
            int verbose):

    cdef int nxi = input.shape[0]
    cdef int nyi = input.shape[1]
    cdef int nzi = input.shape[2]

    cdef int nxo = output.shape[0]
    cdef int nyo = output.shape[1]
    cdef int nzo = output.shape[2]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> input.data
    cdef float *pout = <float *> output.data

    run_mcm_gpu(pinp, pout, iterations, nxi, nyi, nzi, hx, hy, hz, verbose)

def trace_gpu(np.ndarray[float32_t, ndim=3] speed_input,
              np.ndarray[float32_t, ndim=3] output_vol,
              np.ndarray[float32_t, ndim=2] output_trace,
              int x1, int x2, int x3,
              int y1, int y2, int y3,
              int maxstep,
              int verbose):

    cdef int nxi = speed_input.shape[0]
    cdef int nyi = speed_input.shape[1]
    cdef int nzi = speed_input.shape[2]

    cdef int nxo = output_vol.shape[0]
    cdef int nyo = output_vol.shape[1]
    cdef int nzo = output_vol.shape[2]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> speed_input.data
    cdef float *pout_vol = <float *> output_vol.data
    cdef float *pout_trace = <float *> output_trace.data
    cdef int tracelength = 0

    run_trace_gpu(pinp, pout_vol, pout_trace, &tracelength,
                  x1, x2, x3,
                  y1, y2, y3,
                  nxi, nyi, nzi,
                  maxstep,
                  verbose)

    return tracelength