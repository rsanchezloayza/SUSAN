###########################################################################
# This file is part of the Substack Analysis (SUSAN) framework.
# Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
###########################################################################

from . import data
from . import io
from . import utils
from . import modules
from . import project

try:
    from . import ml
except ImportError:
    import warnings as _warnings
    _warnings.warn(
        "susan.ml is not available: PyTorch is not installed. "
        "Install it with:  pip install susan[ml]",
        ImportWarning,
        stacklevel=2,
    )
    del _warnings

def read(filename):
    """Read a SUSAN-supported file and return the appropriate object.

    The file format is inferred from the extension.

    Parameters
    ----------
    filename : str
        Path to the file to read.

    Returns
    -------
    numpy.ndarray or susan.data.Particles or susan.data.Tomograms or susan.data.Reference
        The type depends on the file extension:

        ==================  ===================  ==========================
        Extension(s)        Format               Returned type
        ==================  ===================  ==========================
        ``.mrc`` ``.map``   MRC volume           ``numpy.ndarray``
        ``.ali`` ``.st``
        ``.rec``
        ``.ptclsraw``       SUSAN particles      :class:`susan.data.Particles`
        ``.refstxt``        SUSAN references     :class:`susan.data.Reference`
        ``.tomostxt``       SUSAN tomograms      :class:`susan.data.Tomograms`
        ==================  ===================  ==========================

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    if utils.get_extension(filename) in ('.mrc','.map','.ali','.st','.rec'):
        v,_ = io.mrc.read(filename)
        return v
    elif utils.is_extension(filename,'ptclsraw'):
        return data.Particles(filename)
    elif utils.is_extension(filename,'refstxt'):
        return data.Reference(filename)
    elif utils.is_extension(filename,'tomostxt'):
        return data.Tomograms(filename)
    else:
        raise ValueError('Unsupported file.')

def _check_susan_bin_in_path(bin_name='susan_aligner'):
    from shutil import which
    return which(bin_name) is not None

def _add_susan_bin_to_path(bin_name='susan_aligner'):
    from os.path import dirname,abspath,exists
    import os
    base_dir   = dirname(abspath(__file__))
    local_dir  = abspath(base_dir + '/bin')
    local_file = local_dir+'/'+bin_name
    bin_dir    = abspath(base_dir + '/../bin')
    bin_file   = bin_dir+'/'+bin_name
    build_dir  = abspath(base_dir + '/../build')
    build_file = build_dir+'/'+bin_name
    if exists(local_file):
        os.environ['PATH'] += ':'+local_dir
    elif exists(bin_file):
        os.environ['PATH'] += ':'+bin_dir
    elif exists(build_file):
        os.environ['PATH'] += ':'+build_dir
    else:
        return False
    return True

_HAS_BINARIES = _check_susan_bin_in_path() or _add_susan_bin_to_path()

def has_binaries(bin_name='susan_aligner'):
    """Whether the compiled SUSAN (CUDA) executables are available.

    They are absent when SUSAN was installed without the CUDA modules
    (``SUSAN_NO_CUDA=1 pip install susan``). Reading files and analysing
    results works either way; :mod:`susan.modules` does not.

    Returns
    -------
    bool
    """
    return _check_susan_bin_in_path(bin_name)

def require_binaries(bin_name='susan_aligner'):
    """Raise :class:`RuntimeError` if the compiled executables are missing."""
    if not has_binaries(bin_name):
        raise RuntimeError(
            "'%s' not found: SUSAN was installed without the CUDA modules, or "
            "the binaries are not in the PATH. Reinstall on a machine with the "
            "CUDA toolkit (pip install susan) to enable susan.modules." % bin_name
        )

if not _HAS_BINARIES:
    import warnings as _warnings
    _warnings.warn(
        "The SUSAN executables were not found: susan.modules (alignment, "
        "reconstruction, CTF estimation) is unavailable. Reading and analysing "
        "SUSAN files still works. This is expected if SUSAN was installed with "
        "SUSAN_NO_CUDA=1; otherwise add the binaries to the PATH.",
        UserWarning,
        stacklevel=2,
    )
    del _warnings

__all__ = []
__all__.extend(['read','has_binaries','require_binaries'])
