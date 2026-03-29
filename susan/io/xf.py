__all__ = ['read']

import numpy as _np

def read(filename, parse=True):
    """Read an IMOD alignment transform file (.xf).

    Each line of an .xf file encodes a 2-D affine transform as six values:
    A11 A12 A21 A22 DX DY (linear part followed by translation).

    Parameters
    ----------
    filename : str
        Path to the .xf file.
    parse : bool, optional
        If True (default), reshape the six columns into an array of 2×3
        matrices with shape (N, 2, 3), where each matrix is
        [[A11, A12, DX], [A21, A22, DY]].
        If False, return the raw (N, 6) array as read from the file.

    Returns
    -------
    data : numpy.ndarray
        Shape (N, 2, 3) if parse=True, or (N, 6) if parse=False,
        where N is the number of projections.
    """
    data = _np.loadtxt(filename)
    if parse:
        A11, A12, A21, A22, DX, DY = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        data = _np.stack([_np.stack([A11,A12,DX], axis = -1), _np.stack([A21,A22,DY], axis = -1)], axis = 1)
    return data