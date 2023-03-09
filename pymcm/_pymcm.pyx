import numpy as np

cimport numpy as np

np.import_array()

ctypedef np.float32_t float32_t

cdef extern from "libmcm.h":
    void run_combi_cpu(float *input,
                       float *output,
                       int iterations,
                       float alpha,
                       float beta,
                       int nx, int ny, int nz,
                       int verbose)

    void run_mcm_cpu(float *input,
                     float *output,
                     int iterations,
                     int nx, int ny, int nz,
                     float hx, float hy, float hz,
                     int verbose)

    void run_trace_cpu(float *speed_input,
                       float *output_vol,
                       float *output_trace,
                       int *tracelength,
                       int x1, int x2, int x3,
                       int y1, int y2, int y3,
                       int nx, int ny, int nz,
                       int maxstep,
                       int verbose)

def combi_cpu(np.ndarray[float32_t, ndim=3] input,
              np.ndarray[float32_t, ndim=3] output,
              int iterations,
              float alpha, float beta,
              int verbose):

    cdef int nxi = input.shape[2]
    cdef int nyi = input.shape[1]
    cdef int nzi = input.shape[0]

    cdef int nxo = output.shape[2]
    cdef int nyo = output.shape[1]
    cdef int nzo = output.shape[0]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> input.data
    cdef float *pout = <float *> output.data

    run_combi_cpu(pinp, pout, iterations, alpha, beta, nxi, nyi, nzi, verbose)

def mcm_cpu(np.ndarray[float32_t, ndim=3] input,
            np.ndarray[float32_t, ndim=3] output,
            int iterations,
            float hx, float hy, float hz,
            int verbose):

    cdef int nxi = input.shape[2]
    cdef int nyi = input.shape[1]
    cdef int nzi = input.shape[0]

    cdef int nxo = output.shape[2]
    cdef int nyo = output.shape[1]
    cdef int nzo = output.shape[0]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> input.data
    cdef float *pout = <float *> output.data

    run_mcm_cpu(pinp, pout, iterations, nxi, nyi, nzi, hx, hy, hz, verbose)

def trace_cpu(np.ndarray[float32_t, ndim=3] speed_input,
              np.ndarray[float32_t, ndim=3] output_vol,
              np.ndarray[float32_t, ndim=2] output_trace,
              int x1, int x2, int x3,
              int y1, int y2, int y3,
              int maxstep,
              int verbose):

    cdef int nxi = speed_input.shape[2]
    cdef int nyi = speed_input.shape[1]
    cdef int nzi = speed_input.shape[0]

    cdef int nxo = output_vol.shape[2]
    cdef int nyo = output_vol.shape[1]
    cdef int nzo = output_vol.shape[0]

    assert (nxi == nxo)
    assert (nyi == nyo)
    assert (nzi == nzo)

    cdef float *pinp = <float *> speed_input.data
    cdef float *pout_vol = <float *> output_vol.data
    cdef float *pout_trace = <float *> output_trace.data
    cdef int tracelength = 0

    run_trace_cpu(pinp, pout_vol, pout_trace, &tracelength,
                  x1, x2, x3,
                  y1, y2, y3,
                  nxi, nyi, nzi,
                  maxstep,
                  verbose)

    return tracelength

