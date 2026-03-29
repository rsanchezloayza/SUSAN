__all__ = ['read']

import numpy as _np

def read(filename):
    """Read a IMOD tilt-angle file (.tlt) and return the angles as an array.

    Parameters
    ----------
    filename : str
        Path to the .tlt file.  Each line contains one tilt angle in degrees.

    Returns
    -------
    angles : numpy.ndarray
        1-D array of tilt angles in degrees, one entry per projection.
    """
    return _np.loadtxt(filename)