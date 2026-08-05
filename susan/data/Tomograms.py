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

from __future__ import annotations

import os as _os
import susan.utils.txt_parser as _prsr
import numpy as _np

import susan.io.mrc as _mrc
from susan.io.mrc import get_info as _mrc_info
from susan.io import tlt as _tlt
from susan.io import xf  as _xf

from susan.utils import is_extension as _is_ext
from susan.utils import force_extension as _force_ext
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ
from susan.utils import bin_frame as _bin_frame
from susan.utils import bin_frame_shape as _bin_frame_shape

def lookup_cix(id_table,tomo_id) -> tuple[_np.ndarray, _np.ndarray]:
    """Map tomogram IDs onto their index in ``id_table``.

    Parameters
    ----------
    id_table : ndarray, uint32, shape (N,)
        Tomogram IDs, in container order.  Need not be sorted.
    tomo_id : int or array_like of uint32
        IDs to look up.

    Returns
    -------
    cix : ndarray, int64
        Index into ``id_table``.  Undefined where *valid* is False.
    valid : ndarray of bool
        True where the ID was found.
    """
    ids = _np.asarray(tomo_id,dtype=_np.uint32)
    if id_table.shape[0] == 0:
        return _np.full(ids.shape,-1,dtype=_np.int64),_np.zeros(ids.shape,dtype=bool)
    order = _np.argsort(id_table)
    pos   = _np.searchsorted(id_table[order],ids).clip(0,id_table.shape[0]-1)
    cix   = order[pos].astype(_np.int64)
    valid = (id_table[cix] == ids)
    return cix,valid

class Tomograms:
    """Per-tomogram metadata container for SUSAN workflows.

    Holds geometry, CTF, and acquisition parameters for a set of tomograms.
    Each tomogram entry stores per-projection tilt angles, alignment shifts,
    defocus values, and microscope optics.

    File format: plain-text ``.tomostxt`` (key-value header per tomogram
    followed by one data row per projection).

    .. rubric:: Geometry & acquisition

    .. attribute:: tomo_id
       :type: ndarray, uint32, shape (N,)

       User-assigned integer ID for each tomogram.

    .. attribute:: tomo_size
       :type: ndarray, uint32, shape (N, 3)

       Tomogram dimensions (X, Y, Z) in pixels.

    .. attribute:: tomo_position
       :type: ndarray, float32, shape (N, 3)

       Origin offset (X, Y, Z) of the tomogram in Ångströms, subtracted from
       the particle position before projecting.  Same units as
       :attr:`Particles.position`.  Defaults to (0, 0, 0), which is a no-op.

    .. attribute:: stack_file
       :type: list of str, length N

       Path to the tilt-series image stack for each tomogram.

    .. attribute:: stack_size
       :type: ndarray, uint32, shape (N, 3)

       Stack dimensions (X, Y, n_proj) in pixels.

    .. attribute:: num_proj
       :type: ndarray, uint32, shape (N,)

       Number of valid projections for each tomogram.

    .. attribute:: pix_size
       :type: ndarray, float32, shape (N,)

       Pixel size in Ångströms.

    .. attribute:: proj_eZYZ
       :type: ndarray, float32, shape (N, P, 3)

       Per-projection tilt orientation as ZYZ Euler angles in degrees.

    .. attribute:: proj_shift
       :type: ndarray, float32, shape (N, P, 2)

       Per-projection in-plane shifts (X, Y) in Ångströms.

    .. attribute:: proj_wgt
       :type: ndarray, float32, shape (N, P)

       Per-projection weight (0 = excluded, 1 = active).

    .. attribute:: doses
       :type: ndarray, float32, shape (N, P)

       Cumulative electron dose per projection in e⁻/Ų.

    .. attribute:: nominal_tilt_angles
       :type: ndarray, float32, shape (N, P)

       Stage tilt angles in degrees (from the .tlt file).

    .. rubric:: Optics

    .. attribute:: voltage
       :type: ndarray, float32, shape (N,)

       Accelerating voltage in kV.  Default 300.

    .. attribute:: sph_aber
       :type: ndarray, float32, shape (N,)

       Spherical aberration Cs in mm.  Default 2.7.

    .. attribute:: amp_cont
       :type: ndarray, float32, shape (N,)

       Amplitude contrast fraction.  Default 0.07.

    .. attribute:: handedness
       :type: ndarray, float32, shape (N,)

       Z-axis handedness (+1 or −1).  Default −1.

    .. rubric:: CTF

    .. attribute:: def_U, def_V
       :type: ndarray, float32, shape (N, P)

       Per-projection defocus major/minor axis in Ångströms.

    .. attribute:: def_ang
       :type: ndarray, float32, shape (N, P)

       Defocus astigmatism angle in degrees.

    .. attribute:: def_phas
       :type: ndarray, float32, shape (N, P)

       Phase shift in degrees.

    .. attribute:: def_Bfct
       :type: ndarray, float32, shape (N, P)

       Per-projection B-factor in Å², applied as :math:`e^{-s^2 B/4}`.  Part of
       the CTF model, and therefore *compensated*: the Wiener inversion
       deconvolves it.  Seeds :attr:`susan.data.Particles.def_Bfct`.

    .. attribute:: def_ExFl
       :type: ndarray, float32, shape (N, P)

       Per-projection exposure filter (dose) in Å², applied as
       :math:`e^{-s^2 D/4}`.  Despite sharing its form with :attr:`def_Bfct`,
       it is *uncompensated*: it enters the Wiener numerator only, so it
       persists in the reconstructed map.  Seeds
       :attr:`susan.data.Particles.def_ExFl`, which the aligner then overwrites
       on every run.  See :doc:`/cryoet`.

    .. attribute:: def_mres
       :type: ndarray, float32, shape (N, P)

       Maximum resolution used for CTF fitting in Ångströms.

    .. attribute:: def_scor
       :type: ndarray, float32, shape (N, P)

       CTF fit score.

    .. attribute:: ctf_scale_factor
       :type: ndarray, float32, shape (N, P)

       Per-projection CTF scale factor (RELION convention).
    """

    def __init__(self, filename=None, n_tomo=0, n_proj=0):
        """Load from file or allocate an empty container.

        Parameters
        ----------
        filename : str, optional
            Path to a ``.tomostxt`` file to load.
        n_tomo : int
            Number of tomograms to allocate (used when filename is None).
        n_proj : int
            Maximum number of projections per tomogram (used when filename is None).
        """
        if isinstance(filename, str):
            self._load(filename) 
        else:
            if n_tomo > 0 and n_proj > 0:
                self._alloc(n_tomo,n_proj)
            else:
                raise NameError('Invalid input')
    
    def get_n_tomos(self) -> int:
        """Return the number of tomograms stored."""
        return self.tomo_id.shape[0]

    def get_n_projs(self) -> int:
        """Return the maximum number of projections per tomogram."""
        return self.proj_eZYZ.shape[1]
    
    n_tomos = property(get_n_tomos)
    n_projs = property(get_n_projs)

    def _cix_lookup(self,tomo_id):
        return lookup_cix(self.tomo_id,tomo_id)

    def get_cix(self,tomo_id,strict=True) -> _np.ndarray | int:
        """Map tomogram IDs to their index in this ``Tomograms`` container.

        Parameters
        ----------
        tomo_id : int or array_like of uint32
            Tomogram ID, or array of IDs (e.g. ``Particles.tomo_id``).
        strict : bool, optional
            When True (default), raise if any ID is not present.  When False,
            missing IDs map to -1.

        Returns
        -------
        int or ndarray, int64
            Index into the tomogram arrays, matching the shape of the input.
            Scalar input gives a scalar result.

        Raises
        ------
        ValueError
            If *strict* and one or more IDs are not in this container.
        """
        cix,valid = self._cix_lookup(tomo_id)
        if not valid.all():
            if strict:
                missing = _np.unique(_np.asarray(tomo_id,dtype=_np.uint32)[~valid])
                raise ValueError('tomo_id not found in the tomograms: '
                                 + ','.join(str(m) for m in missing))
            cix = _np.where(valid,cix,-1)
        return cix if cix.ndim > 0 else int(cix)

    def has_tomo(self,tomo_id) -> _np.ndarray | bool:
        """Check which tomogram IDs are present in this container.

        Parameters
        ----------
        tomo_id : int or array_like of uint32
            Tomogram ID, or array of IDs (e.g. ``Particles.tomo_id``).

        Returns
        -------
        bool or ndarray of bool
            True where the ID exists in this container.  Scalar input gives a
            scalar result.
        """
        valid = self._cix_lookup(tomo_id)[1]
        return valid if valid.ndim > 0 else bool(valid)

    def get_is_set(self) -> _np.ndarray:
        """Return a boolean mask of the populated entries.

        An entry is considered *set* once it has a ``stack_file``: that is the
        only field :meth:`__init__` leaves empty, and every way of populating a
        tomogram (:meth:`set_stack`, :meth:`_load`, :meth:`merge_tomos`) fills
        it.  Unset entries are the slots reserved by
        :meth:`add_empty_tomogram_entries` (or by ``Tomograms(n_tomo=...)``)
        that have not been used yet; they are skipped by :meth:`save`.

        Returns
        -------
        ndarray, bool, shape (N,)
        """
        return _np.array([len(str(s)) > 0 for s in self.stack_file],dtype=bool)

    def get_n_set(self) -> int:
        """Return the number of populated entries.  See :meth:`get_is_set`."""
        return int(self.get_is_set().sum())

    def get_next_free_idx(self) -> int:
        """Return the index of the first unset entry, or -1 if there is none.

        Intended to fill a preallocated object without tracking indices::

            i = tomos.next_free_idx
            tomos.set_stack(i,'tomo_001.ali')
        """
        is_set = self.get_is_set()
        if is_set.all():
            return -1
        return int(_np.argmin(is_set))

    is_set        = property(get_is_set)
    n_set         = property(get_n_set)
    next_free_idx = property(get_next_free_idx)

    def _check_unique_ids(self):
        # Unset entries all carry tomo_id=0; they are not saved, so they must
        # not trigger a false collision.
        tomo_id = self.tomo_id[self.get_is_set()]
        uniq,count = _np.unique(tomo_id,return_counts=True)
        if _np.any(count > 1):
            raise ValueError('Repeated tomo_id in the tomograms: '
                             + ','.join(str(u) for u in uniq[count > 1]))

    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'tomostxt'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'tomostxt') + '?')
    
    #def __repr__(self):
    #    return "Tomograms"
    
    def _alloc(self,n_tomos,n_projs):
        self.tomo_id    = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.tomo_size  = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.tomo_position = _np.zeros((n_tomos,3),dtype=_np.float32) # in Angstroms
        self.stack_file = []
        self.stack_size = _np.zeros((n_tomos,3),dtype=_np.uint32 )
        self.num_proj   = _np.zeros( n_tomos   ,dtype=_np.uint32 )
        self.pix_size   = _np.zeros( n_tomos   ,dtype=_np.float32)
        self.proj_eZYZ  = _np.zeros((n_tomos,n_projs,3),dtype=_np.float32)
        self.proj_shift = _np.zeros((n_tomos,n_projs,2),dtype=_np.float32)
        self.proj_wgt   = _np.zeros((n_tomos,n_projs)  ,dtype=_np.float32)
        self.voltage    = 300 *_np.ones( n_tomos,dtype=_np.float32)
        self.sph_aber   = 2.7 *_np.ones( n_tomos,dtype=_np.float32)
        self.amp_cont   = 0.07*_np.ones( n_tomos,dtype=_np.float32)
        self.handedness = (-1.0)*_np.ones( n_tomos,dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_tomos,n_projs),dtype=_np.float32) # score
        
        # Doses
        self.doses = _np.zeros((n_tomos,n_projs),dtype=_np.float32)
        
        # Nominal tilt angles (sorting reasons)
        self.nominal_tilt_angles = _np.zeros((n_tomos,n_projs),dtype=_np.float32)
        
        # CTF Scale Factor (for relion)
        self.ctf_scale_factor = _np.zeros((n_tomos,n_projs),dtype=_np.float32)
        
        for i in range(n_tomos):
            self.stack_file.append('')

    @staticmethod
    def _resolve_stack_path(stack_file,base_dir):
        # If stack_file is a relative path that cannot be found from the current
        # working directory, try to resolve it relative to base_dir (the location
        # of the tomostxt file). Absolute paths are left untouched.
        if _os.path.isabs(stack_file) or _os.path.exists(stack_file):
            return stack_file
        candidate = _os.path.join(base_dir,stack_file)
        if _os.path.exists(candidate):
            return candidate
        return stack_file

    def _load(self,filename):
        Tomograms._check_filename(filename)

        base_dir = _os.path.dirname(filename) or '.'

        fp = open(filename,"rb")
        header  = _prsr.parse_args(fp)
        n_tomos = int(header['num_tomos'])
        n_projs = int(header['num_projs'])
        self._alloc(n_tomos,n_projs)
        for i in range(n_tomos):
            args = _prsr.parse_args(fp)
            self.tomo_id[i]      = _np.uint32(args['tomo_id'])
            self.tomo_size[i,:]  = _np.fromstring(args['tomo_size'],_np.uint32,sep=',')
            if 'tomo_pos' in args:
                self.tomo_position[i,:] = _np.fromstring(args['tomo_pos'],_np.float32,sep=',')
            self.stack_file[i]   = Tomograms._resolve_stack_path(args['stack_file'],base_dir)
            self.stack_size[i,:] = _np.fromstring(args['stack_size'],_np.uint32,sep=',')
            self.pix_size[i]     = _np.float32(args['pix_size'])
            self.voltage[i]      = _np.float32(args['kv'])
            self.sph_aber[i]     = _np.float32(args['cs'])
            self.amp_cont[i]     = _np.float32(args['ac'])
            self.handedness[i]   = _np.float32(args.get('handedness',self.handedness[i]))
            self.num_proj[i]     = _np.uint32(args['num_proj'])

            P = self.num_proj[i]
            for p in range(P):
                buffer = _np.fromstring(_prsr.read_line(fp),dtype=_np.float32,sep=' ')
                self.proj_eZYZ [i,p,:] = buffer[0:3]
                self.proj_shift[i,p,:] = buffer[3:5]
                self.proj_wgt  [i,p]   = buffer[5]
                self.def_U     [i,p]   = buffer[6]
                self.def_V     [i,p]   = buffer[7]
                self.def_ang   [i,p]   = buffer[8]
                self.def_phas  [i,p]   = buffer[9]
                self.def_Bfct  [i,p]   = buffer[10]
                self.def_ExFl  [i,p]   = buffer[11]
                self.def_mres  [i,p]   = buffer[12]
                self.def_scor  [i,p]   = buffer[13]
                if len(buffer) > 14:
                    self.doses[i,p]    = buffer[14]
                if len(buffer) > 15:
                    self.nominal_tilt_angles[i,p] = buffer[15]
                if len(buffer) > 16:
                    self.ctf_scale_factor[i,p] = buffer[16]

        self._check_unique_ids()

    def save(self, filename):
        """Save to a ``.tomostxt`` file.

        Parameters
        ----------
        filename : str
            Output path; must have a ``.tomostxt`` extension.

        Notes
        -----
        Unset entries (see :meth:`get_is_set`) are skipped, so a preallocated
        object can be over-dimensioned and only partially filled.

        Raises
        ------
        ValueError
            If ``tomo_id`` is not unique: the IDs are the key particles are
            matched against, so duplicates cannot be resolved.
        """
        Tomograms._check_filename(filename)
        self._check_unique_ids()

        idx_set = _np.nonzero(self.get_is_set())[0]
        n_skip  = self.n_tomos - idx_set.size
        if n_skip > 0:
            print('[Tomograms.save] %d unset entries skipped (%d saved).'%(n_skip,idx_set.size))

        fp=open(filename,'w')
        _prsr.write(fp,'num_tomos',str(idx_set.size))
        _prsr.write(fp,'num_projs',str(self.n_projs))
        for n,i in enumerate(idx_set):
            fp.write('## Tomogram/Stack '+str(n+1)+'\n')
            _prsr.write(fp,'tomo_id'   , str(self.tomo_id[i]))
            _prsr.write(fp,'tomo_size' , '%d,%d,%d'%(self.tomo_size[i,0],self.tomo_size[i,1],self.tomo_size[i,2]))
            _prsr.write(fp,'tomo_pos'  , '%f,%f,%f'%(self.tomo_position[i,0],self.tomo_position[i,1],self.tomo_position[i,2]))
            _prsr.write(fp,'stack_file', str(self.stack_file[i] ))
            _prsr.write(fp,'stack_size','%d,%d,%d'%(self.stack_size[i,0],self.stack_size[i,1],self.stack_size[i,2]))
            _prsr.write(fp,'pix_size'  , str(self.pix_size[i]))
            _prsr.write(fp,'kv'        , str(self.voltage[i]))
            _prsr.write(fp,'cs'        , str(self.sph_aber[i]))
            _prsr.write(fp,'ac'        , str(self.amp_cont[i]))
            _prsr.write(fp,'handedness', str(self.handedness[i]))
            _prsr.write(fp,'num_proj'  , str(self.num_proj[i]))
            
            fp.write('#euler.Z  euler.Y  euler.Z  shift.X  shift.Y    weight')
            fp.write('  Defocus.U  Defocus.V  Def.ang  PhShift')
            fp.write('  BFactor  ExpFilt')
            fp.write(' Res.angs FitScore\n')
            
            P = self.num_proj[i]
            for p in range(P):
                fp.write('%8.8f %8.8f %8.8f ' % (self.proj_eZYZ[i,p,0],self.proj_eZYZ[i,p,1],self.proj_eZYZ[i,p,2]))
                fp.write('%8.8f %8.8f '       % (self.proj_shift[i,p,0],self.proj_shift[i,p,1]))
                fp.write('%8.8f '             % (self.proj_wgt[i,p]))
                fp.write('%8.8f %8.8f '       % (self.def_U   [i,p],self.def_V   [i,p]))             # Defocus.U    Defocus.V
                fp.write('%8.8f %8.8f '       % (self.def_ang [i,p],self.def_phas[i,p]))             # Def.ang      Def.ph_shft
                fp.write('%8.8f %8.8f '       % (self.def_Bfct[i,p],self.def_ExFl[i,p]))             # Def.BFactor  Def.ExpFilt
                fp.write('%8.8f %8.8f '       % (self.def_mres[i,p],self.def_scor[i,p]))             # Def.max_res  Def.score
                fp.write('%8.8f %8.8f '       % (self.doses[i,p]   ,self.nominal_tilt_angles[i,p]))  # Dose         NominalTiltAngle
                fp.write('%8.8f '             % (self.ctf_scale_factor[i,p]))                        # CTF Scale Factor
                fp.write('\n')
        fp.close()

    # Fields copied verbatim, grouped by shape:
    _FIELDS_TOMO   = ('tomo_id','num_proj','pix_size','voltage','sph_aber',
                            'amp_cont','handedness')
    _FIELDS_TOMO_3 = ('tomo_size','tomo_position','stack_size')
    _FIELDS_PROJ_N = ('proj_eZYZ','proj_shift')
    _FIELDS_PROJ_1 = ('proj_wgt','def_U','def_V','def_ang','def_phas','def_Bfct',
                            'def_ExFl','def_mres','def_scor','doses',
                            'nominal_tilt_angles','ctf_scale_factor')

    @staticmethod
    def merge_tomos(tomos_list, absolute_paths=False) -> Tomograms:
        """Merge several tomogram sets into a new one.

        The per-projection arrays of the inputs are padded to the largest
        ``num_proj`` found, and the tomograms are stored in the order given.
        The ``tomo_id`` values are preserved, so the particles of each input
        remain valid; this requires the IDs to be unique across all inputs.

        Parameters
        ----------
        tomos_list : sequence of Tomograms or str
            Tomogram sets to merge, either already loaded or as paths to
            ``.tomostxt`` files.  The two forms can be mixed.
        absolute_paths : bool
            If True, store ``stack_file`` as absolute paths.  By default the
            paths are kept as they are resolved (relative when the input was
            relative), which keeps the merged file portable as long as it is
            saved next to the inputs.

        Returns
        -------
        Tomograms
            A new object holding all the input tomograms.

        Raises
        ------
        ValueError
            If a ``tomo_id`` appears in more than one input (renumber the
            tomograms and their particles before merging), or if a
            ``stack_file`` cannot be found.

        Examples
        --------
        >>> tomos = susan.data.Tomograms.merge_tomos(['setA.tomostxt',
        ...                                           'setB.tomostxt'])
        >>> tomos.save('merged.tomostxt')
        """
        if isinstance(tomos_list,(str,Tomograms)):
            tomos_list = [tomos_list]

        tomos_in = []
        for entry in tomos_list:
            if isinstance(entry,str):
                tomos_in.append(Tomograms(entry))
            elif isinstance(entry,Tomograms):
                tomos_in.append(entry)
            else:
                raise ValueError('Invalid input: expected a Tomograms object or '
                                 'the path to a .tomostxt file, got '+type(entry).__name__)

        if len(tomos_in) == 0:
            raise ValueError('Empty list of tomograms')

        # Unset entries are ignored: they hold no data and would all collide
        # on tomo_id=0.
        idx_set = [ _np.nonzero(t.get_is_set())[0] for t in tomos_in ]

        # Colliding tomo_ids cannot be resolved here: the particles refer to
        # them, and they are not available at this point.
        all_ids = _np.concatenate( [t.tomo_id[ix] for t,ix in zip(tomos_in,idx_set)] )
        uniq,count = _np.unique(all_ids,return_counts=True)
        if (count>1).any():
            raise ValueError('Repeated tomo_id across the input tomograms: '
                             + str(uniq[count>1])
                             + '. Renumber them (and their particles) before merging.')

        # All the stacks must exist: a merged set pointing to a missing stack
        # only fails much later, inside the GPU modules.
        missing = []
        for t,ix in zip(tomos_in,idx_set):
            for i in ix:
                if not _os.path.exists(t.stack_file[i]):
                    missing.append(t.stack_file[i])
        if len(missing) > 0:
            raise ValueError('Stack file(s) not found: ' + ', '.join(missing))

        n_tomos = sum( ix.size    for ix in idx_set )
        n_projs = max( t.n_projs for t in tomos_in )
        tomos_out = Tomograms(n_tomo=n_tomos,n_proj=n_projs)

        k = 0
        for t,ix in zip(tomos_in,idx_set):
            for i in ix:
                P = int(t.num_proj[i])
                for f in Tomograms._FIELDS_TOMO:
                    getattr(tomos_out,f)[k] = getattr(t,f)[i]
                for f in Tomograms._FIELDS_TOMO_3:
                    getattr(tomos_out,f)[k,:] = getattr(t,f)[i,:]
                for f in Tomograms._FIELDS_PROJ_N:
                    getattr(tomos_out,f)[k,:P,:] = getattr(t,f)[i,:P,:]
                for f in Tomograms._FIELDS_PROJ_1:
                    getattr(tomos_out,f)[k,:P] = getattr(t,f)[i,:P]
                if absolute_paths:
                    tomos_out.stack_file[k] = _os.path.abspath(t.stack_file[i])
                else:
                    tomos_out.stack_file[k] = t.stack_file[i]
                k += 1

        print('[Tomograms.merge_tomos] %d tomograms merged (%d projections max).'%(n_tomos,n_projs))
        print('  Remember to merge the corresponding particles '
              '(Particles.append_ptcls) and to update them with '
              'update_tomo_cix(tomos) before saving.')

        return tomos_out

    def remove_tomograms(self, tid_list):
        """Remove tomograms by ``tomo_id``, in-place.

        The entries matching *tid_list* are dropped from every field.  The
        projection dimension (:attr:`n_projs`) is left untouched, even if the
        widest tomogram was removed.  Unset entries (see :meth:`get_is_set`)
        are never matched, so removing ``0`` does not discard the free slots.

        Parameters
        ----------
        tid_list : int or array_like of uint32
            Tomogram ID, or any sequence of IDs (list, tuple, ndarray).  IDs
            that are not present are reported and ignored.

        Examples
        --------
        >>> tomos.remove_tomograms([3,7])
        >>> tomos.remove_tomograms(12)
        """
        tids = _np.unique(_np.asarray(tid_list,dtype=_np.uint32).reshape(-1))
        if tids.size == 0:
            return

        is_set  = self.get_is_set()
        present = _np.isin(tids,self.tomo_id[is_set])
        if not present.all():
            print('[Tomograms.remove_tomograms] tomo_id not found (ignored): '
                  + ','.join(str(t) for t in tids[~present]))
        if not present.any():
            return

        keep = ~( _np.isin(self.tomo_id,tids[present]) & is_set )
        for f in Tomograms._FIELDS_TOMO + Tomograms._FIELDS_TOMO_3 \
               + Tomograms._FIELDS_PROJ_N + Tomograms._FIELDS_PROJ_1:
            setattr(self,f,getattr(self,f)[keep])
        self.stack_file = [ s for s,k in zip(self.stack_file,keep) if k ]

        print('[Tomograms.remove_tomograms] %d tomograms removed, %d remaining.'
              %(int((~keep).sum()),self.n_tomos))

    def add_empty_tomogram_entries(self, num_entries, n_proj=0):
        """Reserve *num_entries* new (unset) tomogram slots, in-place.

        The new entries are appended at the end and carry the same defaults as
        a freshly allocated object (300 kV, Cs 2.7 mm, 7% amplitude contrast,
        handedness -1) with an empty ``stack_file``, so they are reported as
        unset by :meth:`get_is_set` and skipped by :meth:`save` until they are
        populated (see :meth:`set_stack` and :attr:`next_free_idx`).

        Parameters
        ----------
        num_entries : int
            Number of slots to add.
        n_proj : int, optional
            Minimum number of projections the buffers must hold.  If larger
            than the current :attr:`n_projs`, all the per-projection arrays are
            padded with zeros to that width; the existing data is preserved.

        Examples
        --------
        >>> tomos.add_empty_tomogram_entries(4,n_proj=61)
        >>> tomos.set_stack(tomos.next_free_idx,'tomo_008.ali')
        """
        num_entries = int(num_entries)
        if num_entries < 0:
            raise ValueError('num_entries must be positive')

        n_projs_new = max(self.n_projs,int(n_proj))
        if n_projs_new > self.n_projs:
            pad = n_projs_new - self.n_projs
            for f in Tomograms._FIELDS_PROJ_N:
                arr = getattr(self,f)
                setattr(self,f,_np.pad(arr,((0,0),(0,pad),(0,0))))
            for f in Tomograms._FIELDS_PROJ_1:
                setattr(self,f,_np.pad(getattr(self,f),((0,0),(0,pad))))

        if num_entries == 0:
            return

        # The defaults live in _alloc: build the new entries there and append.
        blank = Tomograms(n_tomo=num_entries,n_proj=n_projs_new)
        for f in Tomograms._FIELDS_TOMO + Tomograms._FIELDS_TOMO_3 \
               + Tomograms._FIELDS_PROJ_N + Tomograms._FIELDS_PROJ_1:
            setattr(self,f,_np.concatenate((getattr(self,f),getattr(blank,f)),axis=0))
        self.stack_file = self.stack_file + blank.stack_file

    @staticmethod
    def _rewrite_paths(tomo_file, out_file, absolute):
        Tomograms._check_filename(tomo_file)
        if out_file is None:
            out_file = tomo_file
        Tomograms._check_filename(out_file)

        tomos   = Tomograms(tomo_file)
        base    = _os.path.abspath(_os.path.dirname(out_file) or '.')
        idx_set = _np.nonzero(tomos.get_is_set())[0]

        missing = []
        for i in idx_set:
            stack = tomos.stack_file[i]
            if not _os.path.exists(stack):
                missing.append(stack)
            stack = _os.path.abspath(stack)
            tomos.stack_file[i] = stack if absolute else _os.path.relpath(stack,base)
        if len(missing) > 0:
            print('[Tomograms] warning, stack file(s) not found: ' + ', '.join(missing))

        tomos.save(out_file)
        return tomos

    @staticmethod
    def relativize_paths(tomo_file, out_file=None) -> Tomograms:
        """Rewrite the ``stack_file`` entries of a ``.tomostxt`` as relative paths.

        The paths are made relative to the directory of the *saved* file, which
        is what :meth:`_load` falls back to when a stack cannot be found from
        the current working directory.  The tomostxt and its stacks can then be
        moved together to another machine.

        Parameters
        ----------
        tomo_file : str
            Path to the ``.tomostxt`` file to rewrite.
        out_file : str, optional
            Where to save the result.  Defaults to *tomo_file* (in-place).

        Returns
        -------
        Tomograms
            The rewritten object, already saved.

        Notes
        -----
        Missing stacks are reported but do not stop the rewrite: the operation
        is purely textual.
        """
        return Tomograms._rewrite_paths(tomo_file,out_file,absolute=False)

    @staticmethod
    def absolutize_paths(tomo_file, out_file=None) -> Tomograms:
        """Rewrite the ``stack_file`` entries of a ``.tomostxt`` as absolute paths.

        Useful when the tomostxt is going to be used from a different working
        directory than the one it was created in.

        Parameters
        ----------
        tomo_file : str
            Path to the ``.tomostxt`` file to rewrite.
        out_file : str, optional
            Where to save the result.  Defaults to *tomo_file* (in-place).

        Returns
        -------
        Tomograms
            The rewritten object, already saved.

        Notes
        -----
        Missing stacks are reported but do not stop the rewrite: the operation
        is purely textual.
        """
        return Tomograms._rewrite_paths(tomo_file,out_file,absolute=True)

    def set_stack(self, idx, stk_name):
        """Populate tomogram entry from a tilt-series stack file.

        Reads the MRC header to fill ``stack_file``, ``stack_size``,
        ``pix_size``, and ``num_proj``.  Projection weights are set to 1
        for all valid projections and 0 for the rest.

        Parameters
        ----------
        idx : int
            Tomogram index to update.
        stk_name : str
            Path to the MRC tilt-series stack.
        """
        stk_dims,apix,_ = _mrc_info(stk_name)
        P = stk_dims[2]
        self.stack_file[idx]   = stk_name
        self.stack_size[idx,:] = stk_dims
        self.pix_size[idx]     = apix[:2].mean()
        self.num_proj[idx]     = P
        self.proj_wgt[idx,:]   = 0
        self.proj_wgt[idx,:P]  = 1
    
    def set_angles(self, idx, tlt_filename, xf_filename=None, xf_apix=None):
        """Set per-projection tilt angles and optional IMOD alignment transforms.

        If only ``tlt_filename`` is provided, tilt angles are set directly
        as the Y Euler angle (ZYZ convention) with no in-plane shifts.
        If ``xf_filename`` is also provided, the IMOD .xf affine transforms
        are combined with the tilt angles to produce full ZYZ orientations
        and X/Y shifts in Ångströms.

        Parameters
        ----------
        idx : int
            Tomogram index to update.
        tlt_filename : str
            Path to a IMOD .tlt file with one tilt angle per line (degrees).
        xf_filename : str, optional
            Path to a IMOD .xf alignment transform file.
        xf_apix : float, optional
            Pixel size to use when converting .xf shifts to Ångströms.
            Defaults to ``pix_size[idx]`` if not given.
        """
        self.proj_eZYZ [idx,:,:] = 0
        self.proj_shift[idx,:,:] = 0
        self.proj_wgt  [idx,:, ] = 0
        
        tlt = _tlt.read(tlt_filename)
        self.nominal_tilt_angles[idx, :self.num_proj[idx]] = tlt
        if (xf_filename == None):
            self.proj_eZYZ[idx, :self.num_proj[idx], 1] = tlt
            self.proj_wgt [idx, :self.num_proj[idx],  ] = 1
        else:
            xf = _xf.read(xf_filename)
            if (xf_apix != None):
                apix = xf_apix
            else:
                apix = self.pix_size[idx]
            for i in range(self.num_proj[idx]):
                rot_tlt = _np.zeros([3,3], _np.float32)
                _euZYZ_rotm(rot_tlt, _np.array([0.0, tlt[i], 0.0], _np.float32) * _np.float32(_np.pi / 180.0))
                rot_xf  = _np.array([[xf[i,0,0], xf[i,0,1], 0],
                                     [xf[i,1,0], xf[i,1,1], 0],
                                     [0        , 0        , 1],
                                    ], _np.float32).T

                vec_xf  = _np.array([xf[i,0,2], xf[i,1,2], 0], _np.float32) * _np.float32(apix)
                vec     = -rot_xf @ vec_xf
                rot     =  rot_xf @ rot_tlt
                euler   = _np.zeros(3, _np.float32)
                _rotm_euZYZ(euler, rot)
                self.proj_eZYZ [idx, i, :] = euler * 180.0 / _np.pi
                self.proj_shift[idx, i, :] = vec[:2]
                self.proj_wgt  [idx, i,  ] = 1
         
         

    def set_defocus(self, idx, def_file, skip_max_res=True):
        """Load CTF defocus parameters from file into a tomogram entry.

        Supports two file formats:

        * ``.defocus`` — IMOD CTFPlotter output (versions 2 and 3).
          Version 2 stores one isotropic defocus per projection; version 3
          stores astigmatic defocus (def_U, def_V, def_ang).
        * ``.txt`` — SUSAN per-projection text format with eight columns:
          def_U, def_V, def_ang, def_phas, def_Bfct, def_ExFl, def_mres, def_scor.

        Projections the CTF estimator flagged as empty (blank frames, written
        out with ``def_U == def_V == 0``) have their projection weight zeroed
        so they are excluded from downstream alignment and reconstruction.

        Parameters
        ----------
        idx : int
            Tomogram index to update.
        def_file : str
            Path to the defocus file (``.defocus`` or ``.txt``).
        skip_max_res : bool, optional
            If True (default), zero out ``def_mres`` after loading so the
            stored maximum-resolution limit is ignored during processing.
        """
        if( _is_ext(def_file,'defocus') ):
            line = _np.loadtxt(def_file,dtype=_np.float32,comments='#',max_rows=1)
            version = int(line[-1])
            if version == 2:
                self.def_U  [idx,0] = 10*line[4]
                self.def_V  [idx,0] = 10*line[4]
                self.def_ang[idx,0] = 0
                data = _np.loadtxt(def_file,dtype=_np.float32,comments='#',skiprows=1)
                n = data.shape[0]
                self.def_U  [idx,1:n+1] = 10*data[:,4]
                self.def_V  [idx,1:n+1] = 10*data[:,4]
                self.def_ang[idx,1:n+1] = 0
            elif version == 3:
                data = _np.loadtxt(def_file,dtype=_np.float32,comments='#',skiprows=1)
                n = data.shape[0]
                self.def_U  [idx,:n] = 10*data[:,4]
                self.def_V  [idx,:n] = 10*data[:,5]
                self.def_ang[idx,:n] = data[:,6]
            else:
                raise NameError('Invalid DEFOCUS format')
        elif _is_ext(def_file,'txt'):
            P = self.num_proj[idx]
            buffer = _np.loadtxt(def_file,dtype=_np.float32,comments='#',ndmin=2,max_rows=P)
            self.def_U     [idx,:P]   = buffer[:,0]
            self.def_V     [idx,:P]   = buffer[:,1]
            self.def_ang   [idx,:P]   = buffer[:,2]
            self.def_phas  [idx,:P]   = buffer[:,3]
            self.def_Bfct  [idx,:P]   = buffer[:,4]
            self.def_ExFl  [idx,:P]   = buffer[:,5]
            self.def_mres  [idx,:P]   = buffer[:,6]
            self.def_scor  [idx,:P]   = buffer[:,7]
            # Optional ninth column: the initial per-projection weight from the
            # Thon ring modulation depth (CtfEstimator.est_initial_snr).  It is
            # proportional to the SSNR of the projection, which is how prj_w
            # enters the reconstruction.  Files written before this column
            # existed simply keep the current proj_wgt.
            if buffer.shape[1] > 8:
                self.proj_wgt[idx,:P] = buffer[:,8]
            # Disable empty projections: the CTF estimator zeroes def_U/def_V
            # (and def_ang, max_res, score) for blank frames in the stack, see
            # mark_empty_projections in src/ctf_linearizer.h.  Mirror that here
            # by zeroing their projection weight so they are excluded from
            # alignment and reconstruction (update_defocus copies proj_wgt into
            # the per-particle prj_w, which the C++/CUDA kernels mask on).
            empty = _np.where((buffer[:,0] == 0) & (buffer[:,1] == 0))[0]
            self.proj_wgt[idx,empty] = 0
            if skip_max_res:
                self.def_mres[idx,:P] = 0
        else:
            raise NameError('Invalid filename')

    def bin(self, scale, in_subfolder=True, filename=None) -> Tomograms:
        """Downsample every tilt-series stack and return a binned ``Tomograms``.

        Each MRC stack referenced in ``stack_file`` is read, every projection
        is downsampled in real space by ``scale`` using a centred area-weighted
        kernel (see :func:`susan.utils.bin_frame`), and the result is written
        to a new MRC alongside the original.  Per-tomogram metadata is updated
        consistently:

        * ``pix_size`` ← ``pix_size * scale``
        * ``stack_size[:, 0:2]`` ← ``ceil(stack_size[:, 0:2] / scale)``
        * ``tomo_size`` ← ``ceil(tomo_size / scale)`` (all three axes)
        * ``stack_file`` ← new path with the ``bSCALE`` tag inserted

        All other fields (tilt angles, shifts in Å, defocus, doses, etc.) are
        binning-invariant and copied verbatim.

        .. note::
           Always bin from the unbinned (b1) stack.  Chaining
           ``b1.bin(2).bin(2)`` is geometrically centre-consistent but applies
           the box kernel twice, which is *not* equivalent to ``b1.bin(4)``
           in the frequency domain.

        Parameters
        ----------
        scale : float, > 1.0
            Downsampling factor.  Integer values are formatted without a
            decimal in the filename tag (``2 → 'b2'``); fractional values use
            ``'p'`` instead of ``'.'`` (``1.5 → 'b1p5'``).
        in_subfolder : bool, optional
            If True (default), each binned stack is written under a
            ``bSCALE/`` sibling directory next to the original stack.
            If False, the binned stack is written in the same directory as
            the original with the tag appended to the stem.
        filename : str, optional
            If given, also save the returned ``Tomograms`` to this
            ``.tomostxt`` file.

        Returns
        -------
        Tomograms
            New ``Tomograms`` instance referencing the binned stacks.
        """
        if scale <= 1.0:
            raise ValueError("scale must be > 1.0")

        s = float(scale)
        tag = ('b%d' % int(s)) if s == int(s) else (('b%g' % s).replace('.', 'p'))
        new = Tomograms(n_tomo=self.n_tomos, n_proj=self.n_projs)

        new.tomo_id[:]              = self.tomo_id
        new.tomo_position[:]        = self.tomo_position # in Angstroms: scale invariant
        new.num_proj[:]             = self.num_proj
        new.pix_size[:]             = self.pix_size * float(scale)
        new.voltage[:]              = self.voltage
        new.sph_aber[:]             = self.sph_aber
        new.amp_cont[:]             = self.amp_cont
        new.handedness[:]           = self.handedness
        new.proj_eZYZ[:]            = self.proj_eZYZ
        new.proj_shift[:]           = self.proj_shift
        new.proj_wgt[:]             = self.proj_wgt
        new.def_U[:]                = self.def_U
        new.def_V[:]                = self.def_V
        new.def_ang[:]              = self.def_ang
        new.def_phas[:]             = self.def_phas
        new.def_Bfct[:]             = self.def_Bfct
        new.def_ExFl[:]             = self.def_ExFl
        new.def_mres[:]             = self.def_mres
        new.def_scor[:]             = self.def_scor
        new.doses[:]                = self.doses
        new.nominal_tilt_angles[:]  = self.nominal_tilt_angles
        new.ctf_scale_factor[:]     = self.ctf_scale_factor

        for i in range(self.n_tomos):
            new.tomo_size[i, 0] = int(_np.ceil(float(self.tomo_size[i, 0]) / float(scale)))
            new.tomo_size[i, 1] = int(_np.ceil(float(self.tomo_size[i, 1]) / float(scale)))
            new.tomo_size[i, 2] = int(_np.ceil(float(self.tomo_size[i, 2]) / float(scale)))

            in_path  = self.stack_file[i]
            in_dir   = _os.path.dirname(in_path)
            in_base  = _os.path.basename(in_path)
            stem, ext = _os.path.splitext(in_base)
            out_base = '%s_%s%s' % (stem, tag, ext if ext else '.mrc')
            if in_subfolder:
                out_dir = _os.path.join(in_dir, tag) if in_dir else tag
                _os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = in_dir
            out_path = _os.path.join(out_dir, out_base) if out_dir else out_base

            stk_in, _ = _mrc.read(in_path)
            P  = int(stk_in.shape[0])
            H  = int(stk_in.shape[1])
            W  = int(stk_in.shape[2])
            H_b, W_b = _bin_frame_shape(H, W, float(scale))

            stk_out = _np.empty((P, H_b, W_b), dtype=_np.float32)
            stk_in_f32 = _np.ascontiguousarray(stk_in, dtype=_np.float32)
            for p in range(P):
                _bin_frame(stk_in_f32[p], float(scale), out_frame=stk_out[p])

            _mrc.write(stk_out, out_path, apix=float(new.pix_size[i]))

            new.stack_file[i]    = out_path
            new.stack_size[i, 0] = W_b
            new.stack_size[i, 1] = H_b
            new.stack_size[i, 2] = P

        if filename is not None:
            new.save(filename)

        return new
