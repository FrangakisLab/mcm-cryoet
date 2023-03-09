import warnings

import numpy as np
from ._pymcm import combi_cpu, mcm_cpu, trace_cpu


try:
    from ._pymcm_cuda import combi_gpu, mcm_gpu, trace_gpu
    GPU_LIB_FOUND = True
except ImportError:
    GPU_LIB_FOUND = False


def mcm_levelset(inp, iterations, alpha, beta, out=None, prefer_gpu=True, verbose=False):
    """
    Smooths a volume using mean curvature and levelset motion.

    Parameters
    ----------
    inp : np.ndarray[float32_t, ndim=3]
        Input image array.
    iterations : int
        Number of iterations
    alpha : float
        amount of level set motion (along surface normals).
    beta : float
        amount of mean curvature motion (along surface curvature)
    out : np.ndarray[float32_t, ndim=3]
        Output image array
    prefer_gpu : bool
        Whether to prefer GPU lib over CPU lib
    verbose : bool
        Whether to print progress

    Returns
    -------

    """
    # Output if necessary
    if out is None:
        out = np.zeros(inp.shape, dtype=np.float32)

    # Size correct (if user supplied)?
    assert(inp.shape == out.shape)

    # Check inputs
    assert(-1 <= alpha <= 1)
    assert(0 <= beta <= 1)

    # GPU lib found?
    if prefer_gpu:
        use_gpu = True
        if not GPU_LIB_FOUND:
            use_gpu = False
            warnings.warn("mcm_levelset: Selected to prefer GPU calculation, but mcm GPU library not found.")
    else:
        use_gpu = False

    # Process image
    if use_gpu:
        if verbose:
            print("Running on GPU.")
        combi_gpu(inp, out, iterations, alpha, beta, verbose)
    else:
        if verbose:
            print("Running on CPU.")
        combi_cpu(inp, out, iterations, alpha, beta, verbose)

    return out


def mcm(inp, iterations, out=None, prefer_gpu=True, hx=1, hy=1, hz=1, verbose=False):
    """
    Smooths a volume using mean curvature motion.

    Parameters
    ----------
     inp : np.ndarray[float32_t, ndim=3]
        Input image array.
    iterations : int
        Number of iterations
    out : np.ndarray[float32_t, ndim=3]
        Output image array
    prefer_gpu : bool
        Whether to prefer GPU lib over CPU lib
    hx : float
        Pixel width in x dim
    hy : float
        Pixel width in y dim
    hz : float
        Pixel width in z dim
    verbose : bool
        Whether to print progress

    Returns
    -------

    """
    # Output if necessary
    if out is None:
        out = np.zeros(inp.shape, dtype=np.float32)

    # Size correct (if user supplied)?
    assert (inp.shape == out.shape)

    # Check inputs
    assert (iterations > 0)
    assert (hx > 0)
    assert (hy > 0)
    assert (hz > 0)

    # GPU lib found?
    if prefer_gpu:
        use_gpu = True
        if not GPU_LIB_FOUND:
            use_gpu = False
            warnings.warn("mcm: Selected to prefer GPU calculation, but mcm GPU library not found.")
    else:
        use_gpu = False

    if use_gpu:
        if verbose:
            print("Running on GPU.")
        mcm_gpu(inp, out, iterations, hx, hy, hz, verbose)
    else:
        if verbose:
            print("Running on CPU.")
        mcm_cpu(inp, out, iterations, hx, hy, hz, verbose)

    return out

def trace(inp, x, y, out_vol=None, out_trace=None, maxstep=10000, prefer_gpu=True, verbose=False):
    """

    Parameters
    ----------
    inp : np.ndarray[float32_t, ndim=3]
        Input speed image array.
    x : vector of 3 ints
        Endpoint of the trace
    y : vector of 3 ints
        Startpoint of the trace
    out_vol : np.ndarray[float32_t, ndim=3]
        Output image array
    out_trace : np.ndarray[float32_t, ndim=2]
        Output trace array
    maxstep : int
        Maximum number of steps to take
    prefer_gpu : bool
        Whether to prefer GPU lib over CPU lib
    verbose : bool
        Whether to print progress

    Returns
    -------

    """
    # Output if necessary
    if out_vol is None:
        out_vol = np.zeros(inp.shape, dtype=np.float32)

    # Trace output if necessary
    if out_trace is None:
        out_trace = np.zeros((maxstep, 3), dtype=np.float32)

    # Size correct (if user supplied)?
    assert (inp.shape == out_vol.shape)
    assert (len(x) == 3)
    assert (len(y) == 3)
    assert (out_trace.shape[0] >= maxstep)
    assert (out_trace.shape[1] == 3)

    # Check inputs
    assert (maxstep > 0)

    # GPU lib found?
    if prefer_gpu:
        use_gpu = True
        if not GPU_LIB_FOUND:
            use_gpu = False
            warnings.warn("trace: Selected to prefer GPU calculation, but mcm GPU library not found.")
    else:
        use_gpu = False

    if use_gpu:
        if verbose:
            print("Running on GPU.")
        tracelen = trace_gpu(inp, out_vol, out_trace,
                             x[0], x[1], x[2],
                             y[0], y[1], y[2],
                             maxstep,
                             verbose)
    else:
        if verbose:
            print("Running on CPU.")
        tracelen = trace_cpu(inp, out_vol, out_trace,
                             x[0], x[1], x[2],
                             y[0], y[1], y[2],
                             maxstep,
                             verbose)

    # Crop trace
    out_trace = out_trace[0:tracelen, :]

    return out_vol, out_trace
