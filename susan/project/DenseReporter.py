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

import os as _os
import json as _json
import math as _math

import numpy as _np

import susan.modules as _ssa_modules
import susan.data as _ssa_data
import susan.io.mrc as _mrc
import susan.utils as _ssa_utils
import susan.utils.datatypes as _dt

__all__ = ['DenseReporter']


class DenseReporter:
    """Dense (template-matching) search over blocks, and the analysis of its output.

    The search runs :class:`susan.modules.Aligner` with ``output_type='dense'``
    (:attr:`aligner`, preset for it): every block (a particle of the input
    file) is searched over all the orientations and its offsets, and the
    aligner writes, per tomogram, a manifest ``{prefix}_t{tomo_id:05d}.json``
    and per block the volumes ``{prefix}_t{tomo_id:05d}_b{ptcl_id:07d}_{cc,z,ang}.mrc``:
    best CC over the orientations, its z-score against the voxel's own
    distribution over the orientations, and the index of the best orientation
    in the manifest's angle table.  The pose of a voxel is
    R(ali_eu of its block) R(table[ang]).

    The blocks are temporary: :meth:`run` ends with :meth:`finalize`, which
    assembles the tomogram volumes ``{prefix}_t{tomo_id:05d}_{z,cc,ang}.mrc``
    (where blocks overlap, the sample with the highest z, with its cc and
    angle), records them in the manifest and deletes the block files (unless
    :attr:`keep_blocks`).  Voxels no block searched hold 0 (z, cc) and -1
    (ang); the header statistics are those of the searched voxels.  Blocks
    with their own ``ali_eu`` (a local search from posed particles) also get
    ``_blk.mrc``, the index of the winning block in the manifest.

    The analysis (:meth:`histogram`, :meth:`assemble`, :meth:`coverage`,
    :meth:`pick` and the exports) uses the loaded manifests: :meth:`run` loads
    its own, :meth:`load` loads earlier ones.

    :meth:`run_null` searches a few random slab blocks with a phase-randomized
    decoy of the reference (same amplitudes and mask, random phases): the
    null distribution of z with the data and geometry intact but nothing to
    match.  When a null is loaded, :meth:`threshold` and :meth:`pick` use it;
    otherwise they use the robust statistics of the whole slab.

    Lengths (:attr:`offset_xy`, :attr:`offset_z`, :attr:`offset_step`,
    :attr:`thickness`, the pick radius) are in pixels of the tomograms; use
    :meth:`A2pix` for values in angstroms.  :attr:`slab` is in angstroms.

    Attributes
    ----------
    aligner : :class:`susan.modules.Aligner`
        The search settings (bandpass, angular sampling, dilation, CTF and
        normalization).  :meth:`run` sets the output type, dimensionality,
        refinement, offsets and binning.
    tomograms : str
        ``.tomostxt`` file.  Setting it reads the tomograms and
        :attr:`pix_size`; all tomograms must share one pixel size.
    reference : str
        ``.refstxt`` file with one reference.
    box_size : int
        Box size of the search, in pixels.
    offset_xy, offset_z : int or ``'auto'``
        Half-size of the offsets searched per block, in pixels.  ``'auto'``:
        offset_z is half of :attr:`thickness` (half of the tomogram thickness
        without slab and thickness, at most half of the box budget), and
        offset_xy the largest r with sqrt(r^2 + offset_z^2) + R <= box/2, R
        the particle radius from the reference mask.  Default ``'auto'``.
    offset_step : float
        Step of the offsets, in pixels (sub-voxel steps are allowed).
        Default ``1``.
    thickness : float or None
        Thickness searched, in pixels, centred on the slab plane (or on the
        tomogram centre without slab).  ``None``: the whole tomogram without
        slab, one layer of blocks with slab.  Default ``None``.
    slab : tuple (a, b, c), dict {tomo_id: (a, b, c)} or None
        Specimen plane, Z = a X + b Y + c, in angstroms, in the particle frame
        (relative to the tomogram centre).  Default ``None``.
    dense_bin : int
        Binning of the output volumes (done in the aligner).  Default ``1``.
    dense_reduce : str
        ``'max'`` or ``'mean'``: reduction of the samples of an output voxel.
    keep_blocks : bool
        Keep the block files after :meth:`finalize` (debugging).  Default
        ``False``.
    list_gpus_ids, mpi, verbosity
        As in :class:`susan.project.SubtomoAvg`.
    manifests : dict
        Loaded manifests, {tomo_id: dict}.
    """

    def __init__(self, json_file=None):
        self.aligner = _ssa_modules.Aligner()
        self.aligner.output_type    = 'dense'
        self.aligner.dimensionality = 3
        self.aligner.refine         = _dt.refine_params(0,2)
        self.aligner.offset_space   = 'tomogram'
        self.aligner.cc_type        = 'cfsc'
        self.aligner.ctf_correction = 'on_reference'
        self.aligner.normalize_type = 'zero_mean_one_std'
        self.aligner.dilate         = 1.5

        self.list_gpus_ids = [0]
        self.mpi           = _dt.mpi_params('srun -n %d ',1)
        self.verbosity     = 0

        self.tomogram_file = None
        self.pix_size      = None
        self._tomos        = None
        self.reference     = None
        self.box_size      = None
        self.offset_xy     = 'auto'
        self.offset_z      = 'auto'
        self.offset_step   = 1.0
        self.thickness     = None
        self.slab          = None
        self.dense_bin     = 1
        self.dense_reduce  = 'max'
        self.keep_blocks   = False

        self.manifests     = {}
        self.null_manifests = {}
        self._prefix       = None
        if json_file is not None:
            self.load(json_file)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @property
    def tomograms(self):
        return self.tomogram_file

    @tomograms.setter
    def tomograms(self, filename):
        T = _ssa_data.Tomograms(filename)
        ps = _np.unique(T.pix_size)
        if len(ps) > 1:
            raise ValueError('The tomograms must share one pixel size (found %s).' % ', '.join('%.4f' % p for p in ps))
        self.tomogram_file = filename
        self._tomos        = T
        self.pix_size      = float(ps[0])

    def A2pix(self, angstroms):
        """Length in angstroms to pixels of the tomograms (:attr:`pix_size`)."""
        self._need_pix_size()
        return angstroms/self.pix_size

    def pix2A(self, pixels):
        """Length in pixels of the tomograms to angstroms."""
        self._need_pix_size()
        return pixels*self.pix_size

    def A2fpix(self, angstroms):
        """Resolution in angstroms to Fourier pixels of the box: box_size * pix_size / angstroms."""
        self._need_pix_size()
        return self.box_size*self.pix_size/angstroms

    def fpix2A(self, fpix):
        """Radius in Fourier pixels of the box to a resolution in angstroms."""
        self._need_pix_size()
        return self.box_size*self.pix_size/fpix

    def _need_pix_size(self):
        if self.pix_size is None:
            raise ValueError('Set tomograms first: the pixel size comes from them.')

    def particle_radius(self):
        """Radius (pixels) of the sphere with the volume of the reference mask."""
        ref = _ssa_data.Reference(self.reference)
        m,_ = _mrc.read(ref.msk[0])
        return float(_np.cbrt(3*_np.clip(m,0,1).sum()/(4*_np.pi)))

    def _slab_of(self, tomo_id):
        if self.slab is None:
            return None
        if isinstance(self.slab,dict):
            return self.slab[tomo_id]
        return self.slab

    @staticmethod
    def _slab_z0(slab, x, y, size, apix):
        """Height (pixels) of the slab plane at the pixel positions x, y; the tomogram centre without slab."""
        if slab is None:
            return _np.full(_np.broadcast(x,y).shape,size[2]/2)
        return (slab[0]*(x - size[0]/2)*apix + slab[1]*(y - size[1]/2)*apix + slab[2])/apix + size[2]/2

    def _cix(self, tomo_id):
        return int(self._tomos.get_cix(tomo_id))

    def resolved(self, tomo_id=None):
        """Offsets, spacing and layers (pixels) that :meth:`run` uses, for the given tomograms.

        Returns a dict; one aligner run uses the same offsets for every block,
        so ``'auto'`` values are the smallest over the tomograms."""
        if self._tomos is None or self.box_size is None:
            raise ValueError('Set tomograms and box_size first.')
        tids = self._tomo_ids(tomo_id)
        b    = int(self.dense_bin)
        budget = self.box_size/2.0
        note = []
        if self.offset_xy == 'auto' or self.offset_z == 'auto':
            R = self.particle_radius()
            budget -= R
            note.append('R %.1f px' % R)
        rz_all, depth = [], {}
        for t in tids:
            cix  = self._cix(t)
            size = self._tomos.tomo_size[cix].astype(float)
            if self.thickness is not None:
                depth[t] = float(self.thickness)
            elif self._slab_of(t) is None:
                depth[t] = size[2]
            else:
                depth[t] = None
            if self.offset_z == 'auto':
                if depth[t] is None:
                    raise ValueError('offset_z "auto" with a slab needs thickness.')
                rz_all.append(min(_math.ceil(depth[t]/2), _math.floor(budget/2)))
        rz = int(min(rz_all)) if self.offset_z == 'auto' else int(self.offset_z)
        if self.offset_xy == 'auto':
            if budget <= rz:
                raise ValueError('The box is too small for offset_z %d (budget %.1f px).' % (rz,budget))
            rxy = int(_math.floor(_math.sqrt(budget**2 - rz**2)))
            note.append('auto: sqrt(r_xy^2 + r_z^2) + R <= box/2')
        else:
            rxy = int(self.offset_xy)
        s_xy = max(b*((2*rxy+1)//b),b)
        s_z  = max(b*((2*rz +1)//b),b)
        layers = {t: (1 if depth[t] is None else max(int(_math.ceil(depth[t]/(2*rz+1))),1)) for t in tids}
        return dict(offset_xy=rxy, offset_z=rz, offset_step=float(self.offset_step), spacing_xy=s_xy, spacing_z=s_z,
                    layers=layers, dense_bin=b, note=', '.join(note))

    # ------------------------------------------------------------------
    # Grid and run
    # ------------------------------------------------------------------

    def _tomo_ids(self, tomo_id):
        if tomo_id is None:
            return [int(t) for t in self._tomos.tomo_id]
        if _np.isscalar(tomo_id):
            return [int(tomo_id)]
        return [int(t) for t in tomo_id]

    @staticmethod
    def _axis(L, r, s):
        """Centres from r to L-1-r, evenly spread with spacing <= s: no box extends past the tomogram."""
        if L - 1 <= 2*r:
            return _np.array([(L - 1)/2.0])
        n = int(_math.ceil((L - 1 - 2*r)/float(s))) + 1
        return _np.linspace(r, L - 1 - r, n)

    @staticmethod
    def _snap(p, size, b):
        """Nearest multiples of b, kept inside [0, size - 1]."""
        p = b*_np.sign(p/b)*_np.floor(_np.abs(p/b) + 0.5)
        p = _np.where(p > size - 1, p - b, p)
        return _np.where(p < 0, p + b, p)

    def _blocks(self, pos, tid):
        """Particles at the given block centres (voxels from the tomogram corner)."""
        pos = _np.concatenate(pos).astype(_np.float32)
        tid = _np.concatenate(tid)
        P = _ssa_data.Particles.import_data(position=pos, tomograms=self._tomos, tomos_id=tid)
        for t in _np.unique(tid):
            P.position[P.tomo_id == t] += self._tomos.tomo_position[self._cix(t)]
        if _np.any(self._tomos.tomo_position):
            P.update_defocus(self._tomos)
        return P

    def _point_blocks(self, points, tomo_id, one_based=False):
        """Particles at user points: one (N, 3) set for every tomogram, or one set per tomogram matched
        with a tomo_id list.  Points outside the tomogram and duplicates after the snapping are dropped."""
        tids = self._tomo_ids(tomo_id)
        try:
            arr = _np.asarray(points, dtype=float)
        except ValueError:
            arr = None
        if arr is not None and arr.ndim <= 2:
            arr  = arr.reshape(-1, 3) if arr.size else _np.zeros((0, 3))
            sets = [arr]*len(tids)
        else:
            if tomo_id is None or _np.isscalar(tomo_id):
                raise ValueError('Per-tomogram points need a tomo_id list of the same length.')
            sets = [_np.asarray(q, dtype=float).reshape(-1, 3) for q in points]
            if len(sets) != len(tids):
                raise ValueError('%d point sets for %d tomograms.' % (len(sets), len(tids)))
        if one_based:
            sets = [q - 1 for q in sets]
        b = int(self.dense_bin)
        pos, tid = [], []
        for t, q in zip(tids, sets):
            size = self._tomos.tomo_size[self._cix(t)].astype(float)
            inside = _np.all((q >= 0) & (q <= size - 1), axis=1)
            p = _np.unique(self._snap(q[inside], size, b), axis=0)
            if inside.sum() < len(q) or len(p) < inside.sum():
                print('TomoID %d: %d points, %d outside the tomogram and %d duplicates dropped.' % (
                    t, len(q), len(q) - inside.sum(), inside.sum() - len(p)))
            if len(p):
                pos.append(p)
                tid.append(_np.full(len(p), t, dtype=_np.uint32))
        if not pos:
            raise ValueError('No points inside the tomograms.')
        return self._blocks(pos, tid)

    def make_grid(self, tomo_id=None):
        """Block centres for the given tomograms (all by default), as :class:`susan.data.Particles`.

        XY: centres evenly spread from offset_xy to size - 1 - offset_xy, at
        most the :attr:`resolved` spacing apart, so the searched region ends at
        the tomogram edges and no block hangs over them; Z: the layers around
        the slab plane (or the tomogram centre).  Centres are
        multiples of :attr:`dense_bin` (voxels from the tomogram corner)."""
        r    = self.resolved(tomo_id)
        b    = r['dense_bin']
        pos, tid = [], []
        for t in self._tomo_ids(tomo_id):
            cix  = self._cix(t)
            size = self._tomos.tomo_size[cix].astype(float)
            apix = float(self._tomos.pix_size[cix])
            ax_x = self._axis(size[0], r['offset_xy'], r['spacing_xy'])
            ax_y = self._axis(size[1], r['offset_xy'], r['spacing_xy'])
            x, y = _np.meshgrid(ax_x, ax_y, indexing='xy')
            x, y = x.ravel(), y.ravel()
            z0 = self._slab_z0(self._slab_of(t), x, y, size, apix)
            n = r['layers'][t]
            for i in range(n):
                z = z0 + (i - (n-1)/2.0)*r['spacing_z']
                pos.append(self._snap(_np.column_stack((x, y, z)), size, b))
                tid.append(_np.full(len(x), t, dtype=_np.uint32))
        return self._blocks(pos, tid)

    def run(self, output_prefix, ptcls=None, tomo_id=None, points=None, one_based=False, imod_coords=None):
        """Run the dense search and load its manifests.

        Parameters
        ----------
        output_prefix : str
            Prefix of the output files; its folders are created.
        ptcls : str or Particles, optional
            Blocks to search.  Default: :meth:`make_grid`.
        tomo_id : int or list of int, optional
            Tomograms to search (all by default).
        points : array_like, optional
            Block centres instead of ``ptcls``, in voxels from the tomogram
            corner (x, y, z).  An (N, 3) array is searched in every
            tomogram of ``tomo_id``; a list of (N_i, 3) arrays (or an
            (n, N, 3) array) is matched with a ``tomo_id`` list of the same
            length.  Centres are snapped to multiples of :attr:`dense_bin`;
            points outside the tomogram and duplicates are dropped.
        one_based : bool
            ``points`` count from 1, as displayed by 3dmod; default ``False``
            (0-based, as :meth:`pick` and :meth:`to_csv`).
        imod_coords : bool, optional
            Alias of ``one_based``.

        Returns
        -------
        str or dict
            The manifest for a single ``tomo_id``, otherwise {tomo_id: manifest}.
        """
        if self.reference is None:
            raise ValueError('Set reference.')
        r = self.resolved(tomo_id)
        if points is not None and ptcls is not None:
            raise ValueError('Give ptcls or points, not both.')
        if imod_coords is not None:
            one_based = bool(imod_coords)
        if points is not None:
            P = self._point_blocks(points, tomo_id, one_based)
        elif ptcls is None:
            P = self.make_grid(tomo_id)
        else:
            P = _ssa_data.Particles(ptcls) if isinstance(ptcls,str) else ptcls
            if tomo_id is not None:
                P = P.select(_np.isin(P.tomo_id, self._tomo_ids(tomo_id)))
        tids = [int(t) for t in _np.unique(P.tomo_id)]
        self._run_aligner(output_prefix, self.reference, P, r)
        self._prefix = output_prefix
        files = {t: '%s_t%05d.json' % (output_prefix,t) for t in tids}
        self.load(files)
        self.finalize(tids)
        if tomo_id is not None and _np.isscalar(tomo_id):
            return files[int(tomo_id)]
        return files

    def _run_aligner(self, output_prefix, refs_file, P, r):
        d = _os.path.dirname(output_prefix)
        if d and not _os.path.isdir(d):
            _os.makedirs(d)
        ptcls_file = output_prefix + '_blocks.ptclsraw'
        P.save(ptcls_file)
        a = self.aligner
        a.output_type    = 'dense'
        a.dimensionality = 3
        a.refine         = _dt.refine_params(0,a.refine.factor)
        a.offset_space   = 'tomogram'
        a.offset         = _dt.offset_params([r['offset_xy'],r['offset_xy'],r['offset_z']],r['offset_step'],'cuboid')
        a.dense_bin      = r['dense_bin']
        a.dense_reduce   = self.dense_reduce
        a.list_gpus_ids  = self.list_gpus_ids
        a.verbosity      = self.verbosity
        if self.mpi.arg > 1:
            a.mpi.cmd = self.mpi.cmd
            a.mpi.arg = self.mpi.arg
            a.align_mpi(output_prefix, refs_file, self.tomogram_file, ptcls_file, self.box_size)
        else:
            a.align(output_prefix, refs_file, self.tomogram_file, ptcls_file, self.box_size)

    def make_decoy(self, folder, seed=0):
        """Phase-randomized decoy of :attr:`reference` in ``folder``: :func:`susan.utils.phase_randomize` of
        the map (all frequencies), times the reference mask, scaled to the map's mean / std inside the mask.
        Returns the decoy ``.refstxt``."""
        ref = _ssa_data.Reference(self.reference)
        vol, apix = _mrc.read(ref.ref[0])
        msk, _ = _mrc.read(ref.msk[0])
        vol = vol.astype(_np.float64)
        msk = _np.clip(msk.astype(_np.float64),0,1)
        dec = _ssa_utils.phase_randomize(vol,fpix=0,seed=seed).astype(_np.float64)*msk
        inm = msk > 0.5
        dec = ((dec - dec[inm].mean())/dec[inm].std()*vol[inm].std() + vol[inm].mean())*msk
        map_file = _os.path.join(folder,'decoy.mrc')
        _mrc.write(dec.astype(_np.float32),map_file,apix=float(apix[0]))
        refs_file = _os.path.join(folder,'decoy.refstxt')
        with open(refs_file,'w') as fp:
            fp.write('num_ref:1\n## Reference 1\nmap:%s\nmask:%s\nh1:\nh2:\n' % (_os.path.abspath(map_file),_os.path.abspath(ref.msk[0])))
        return refs_file

    def run_null(self, output_prefix=None, n=None, tomo_id=None, seed=0):
        """Null of the search: ``n`` blocks per tomogram at random positions on the slab, searched with
        a phase-randomized decoy of the reference (written to a temporary folder, removed after), with the
        same settings as :meth:`run`.  Writes ``{prefix}_t{tomo_id:03d}.json`` (marked ``"null": "decoy"``)
        and loads them.

        Parameters
        ----------
        output_prefix : str, optional
            Default: the prefix of the last :meth:`run` + ``'_null'``.
        n : int, optional
            Blocks per tomogram.  Default: max(2, number of GPUs of the run),
            i.e. ``len(list_gpus_ids)`` times the MPI processes.
        tomo_id : int or list of int, optional
            Default: the loaded tomograms, or all.
        seed : int
            Seed of the positions and of the decoy phases.

        Returns
        -------
        str or dict
            As :meth:`run`.
        """
        import tempfile, shutil
        if output_prefix is None:
            if self._prefix is None:
                raise ValueError('Give output_prefix (no run() in this session).')
            output_prefix = self._prefix + '_null'
        if self.reference is None:
            raise ValueError('Set reference.')
        if n is None:
            n = max(2, len(self.list_gpus_ids)*max(int(self.mpi.arg),1))
        if tomo_id is None and self.manifests:
            tomo_id = sorted(self.manifests)
        r    = self.resolved(tomo_id)
        b    = r['dense_bin']
        rng  = _np.random.default_rng(seed)
        pos, tid = [], []
        for t in self._tomo_ids(tomo_id):
            cix  = self._cix(t)
            size = self._tomos.tomo_size[cix].astype(float)
            apix = float(self._tomos.pix_size[cix])
            x = rng.uniform(r['offset_xy'],size[0]-1-r['offset_xy'],n)
            y = rng.uniform(r['offset_xy'],size[1]-1-r['offset_xy'],n)
            z0 = self._slab_z0(self._slab_of(t),x,y,size,apix)
            half = max(r['layers'][t]*r['spacing_z']/2.0 - r['offset_z'] - 0.5,0.0)
            z = z0 + rng.uniform(-half,half,n)
            pos.append(self._snap(_np.column_stack((x,y,z)),size,b))
            tid.append(_np.full(n,t,dtype=_np.uint32))
        P   = self._blocks(pos,tid)
        tmp = tempfile.mkdtemp(prefix='susan_decoy_')
        try:
            self._run_aligner(output_prefix, self.make_decoy(tmp,seed), P, r)
        finally:
            shutil.rmtree(tmp,ignore_errors=True)
        tids  = [int(t) for t in _np.unique(tid)]
        files = {t: '%s_t%05d.json' % (output_prefix,t) for t in tids}
        for f in files.values():
            with open(f) as fp:
                m = _json.load(fp)
            m['null'] = 'decoy'
            m['params']['refs_file'] = self.reference
            with open(f,'w') as fp:
                _json.dump(m,fp,indent=1)
        self.load(files)
        self.finalize(tids)
        if _np.isscalar(tomo_id):
            return files[int(tomo_id)]
        return files

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, json_file):
        """Load manifests (search or null, recognized from the file): a path, a list or a {tomo_id: path} dict."""
        if isinstance(json_file,dict):
            json_file = list(json_file.values())
        if isinstance(json_file,str):
            json_file = [json_file]
        for f in json_file:
            with open(f) as fp:
                m = _json.load(fp)
            m['_file'] = f
            base = _os.path.dirname(f)
            for k,v in m.get('volumes',{}).items():
                if not _os.path.exists(v):
                    cand = _os.path.join(base,_os.path.basename(v))
                    if _os.path.exists(cand):
                        m['volumes'][k] = cand
            for blk in m['blocks']:
                blk['ptcl_id'] %= 2**32
                for k,v in blk['files'].items():
                    if not _os.path.exists(v):
                        cand = _os.path.join(base,_os.path.basename(v))
                        if _os.path.exists(cand):
                            blk['files'][k] = cand
                blk['_lo'] = _np.array(blk['corner_xyz'],dtype=int)[::-1]
                blk['_hi'] = blk['_lo'] + _np.array(blk['shape_xyz'],dtype=int)[::-1]
            m['_table'] = _np.array(m['angle_table'],dtype=_np.float32)
            if 'null' in m:
                self.null_manifests[int(m['tomogram']['tomo_id'])] = m
            else:
                self.manifests[int(m['tomogram']['tomo_id'])] = m
        if self._tomos is None and (self.manifests or self.null_manifests):
            m = next(iter((self.manifests or self.null_manifests).values()))
            if _os.path.exists(m['params']['tomos_file']):
                self.tomograms = m['params']['tomos_file']

    def _manifest(self, tomo_id):
        if not self.manifests:
            raise ValueError('No manifest loaded: run() or load() first.')
        if tomo_id is None:
            if len(self.manifests) > 1:
                raise ValueError('Several tomograms loaded (%s): give tomo_id.' % sorted(self.manifests))
            tomo_id = next(iter(self.manifests))
        return int(tomo_id), self.manifests[int(tomo_id)]

    @staticmethod
    def _read(blk, field):
        f = blk['files'].get(field)
        if f is None or not _os.path.exists(f):
            return _np.full(tuple(blk['shape_xyz'][::-1]),_np.nan,dtype=_np.float32)
        v,_ = _mrc.read(f)
        return v

    @staticmethod
    def _need_volumes(m):
        if 'volumes' not in m and 'hist' not in m:
            raise ValueError('Tomogram %d is not assembled: call finalize() first.' % m['tomogram']['tomo_id'])

    @staticmethod
    def _read_vol(m, field):
        """Stored volume; z and cc are NaN where no block searched (ang < 0)."""
        if field not in m['volumes']:
            raise ValueError('No %s volume (written only for blocks with their own ali_eu).' % field)
        v,_ = _mrc.read(m['volumes'][field])
        if field in ('z','cc'):
            ang,_ = _mrc.read(m['volumes']['ang'])
            v = _np.where(ang >= 0,v,_np.nan).astype(_np.float32)
        return v

    @staticmethod
    def _save_manifest(m):
        out = {k:v for k,v in m.items() if not k.startswith('_')}
        out['blocks'] = [{k:v for k,v in blk.items() if not k.startswith('_')} for blk in m['blocks']]
        with open(m['_file'],'w') as fp:
            _json.dump(out,fp,indent=1)

    def _assemble_blocks(self, m):
        """One pass over the blocks: z, cc, ang and blk (block index, -1 where none) of the highest-z sample."""
        shape = tuple(m['tomogram']['size_out_xyz'][::-1])
        vol = dict(z=_np.full(shape,-_np.inf,dtype=_np.float32), cc=_np.full(shape,_np.nan,dtype=_np.float32),
                   ang=_np.full(shape,-1,dtype=_np.float32), blk=_np.full(shape,-1,dtype=_np.float32))
        for i,blk in enumerate(m['blocks']):
            lo, hi = blk['_lo'], blk['_hi']
            a = _np.maximum(lo,0)
            e = _np.minimum(hi,shape)
            if _np.any(e <= a):
                continue
            src = tuple(slice(a[j]-lo[j],e[j]-lo[j]) for j in range(3))
            dst = tuple(slice(a[j],e[j]) for j in range(3))
            z = self._read(blk,'z')[src]
            upd = z > vol['z'][dst]
            if not upd.any():
                continue
            vol['z'][dst]   = _np.where(upd,z,vol['z'][dst])
            vol['cc'][dst]  = _np.where(upd,self._read(blk,'cc')[src],vol['cc'][dst])
            vol['ang'][dst] = _np.where(upd,self._read(blk,'ang')[src],vol['ang'][dst])
            vol['blk'][dst] = _np.where(upd,_np.float32(i),vol['blk'][dst])
        vol['z'][_np.isinf(vol['z'])] = _np.nan
        return vol

    def finalize(self, tomo_id=None, keep_blocks=None):
        """Replace the block outputs of the loaded manifests by their assembly.

        Search manifests: the volumes ``{json stem}_{z,cc,ang,blk}.mrc`` (see
        the class description).  Null manifests: the histograms and
        statistics of z and cc (as :meth:`histogram`).  Both are recorded in
        the manifest and the block files are deleted unless ``keep_blocks``
        (default :attr:`keep_blocks`).  Finalized manifests are skipped.
        """
        keep = self.keep_blocks if keep_blocks is None else keep_blocks
        tids = None if tomo_id is None else set(self._tomo_ids(tomo_id))
        for null,mans in ((False,self.manifests),(True,self.null_manifests)):
            for t,m in mans.items():
                if (tids is not None and t not in tids) or ('hist' if null else 'volumes') in m:
                    continue
                if null:
                    m['hist'] = {}
                    for fld in ('z','cc'):
                        counts, edges, st = self._histogram(lambda fld=fld,m=m: (self._read(b,fld) for b in m['blocks']))
                        m['hist'][fld] = dict(counts=counts.tolist(), edges=edges.tolist(), stats=st)
                else:
                    stem = _os.path.splitext(m['_file'])[0]
                    vols = self._assemble_blocks(m)
                    ok = vols['ang'] >= 0
                    if not any(_np.any(blk['ali_eu']) for blk in m['blocks']):
                        del vols['blk']
                    m['volumes'] = {}
                    for fld,v in vols.items():
                        p = '%s_%s.mrc' % (stem,fld)
                        v = _np.where(ok,v,0 if fld in ('z','cc') else -1).astype(_np.float32)
                        _mrc.write(v,p,apix=m['tomogram']['apix_out'],fill_statistics=False)
                        _mrc.set_statistics(p,v,ok)
                        m['volumes'][fld] = p
                if not keep:
                    for blk in m['blocks']:
                        for p in blk['files'].values():
                            if _os.path.exists(p):
                                _os.remove(p)
                        blk['files'] = {}
                self._save_manifest(m)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def histogram(self, field='z', tomo_id=None, bins=200, value_range=None, source='blocks'):
        """Histogram of ``'z'`` or ``'cc'`` over the searched voxels of a tomogram (NaN skipped).

        source ``'blocks'`` (the search, its assembled volume) or ``'null'``
        (the stored histogram of the :meth:`run_null` blocks; ``bins`` and
        ``value_range`` are ignored).  Returns (counts, edges, stats); stats
        holds n, mean, std, median, sigma = 1.4826 MAD, p99, p99.9 and max
        (from the histogram); median and sigma are the robust null of
        :meth:`threshold`."""
        tid, m = self._manifest(tomo_id)
        if source == 'null':
            m = self.null_manifests.get(tid)
            if m is None:
                raise ValueError('No null loaded for tomogram %d.' % tid)
            self._need_volumes(m)
            h = m['hist'][field]
            return _np.array(h['counts'],dtype=_np.int64), _np.array(h['edges']), dict(h['stats'])
        self._need_volumes(m)
        return self._histogram(lambda: [self._read_vol(m,field)],bins,value_range)

    @staticmethod
    def _histogram(chunks, bins=200, value_range=None):
        """Histogram and statistics of the finite values of the arrays from ``chunks()`` (called twice without range)."""
        if value_range is None:
            lo, hi = _np.inf, -_np.inf
            for v in chunks():
                if _np.isfinite(v).any():
                    lo, hi = min(lo,_np.nanmin(v)), max(hi,_np.nanmax(v))
            value_range = (float(lo),float(hi))
        counts = _np.zeros(bins,dtype=_np.int64)
        s = s2 = 0.0
        for v in chunks():
            v = v[_np.isfinite(v)]
            counts += _np.histogram(v,bins=bins,range=value_range)[0]
            s  += float(v.sum(dtype=_np.float64))
            s2 += float((v.astype(_np.float64)**2).sum())
        edges = _np.linspace(value_range[0],value_range[1],bins+1)
        n = int(counts.sum())
        centres = 0.5*(edges[1:] + edges[:-1])
        cdf = _np.cumsum(counts)/max(n,1)
        median = float(_np.interp(0.5,cdf,centres))
        dev = _np.abs(centres - median)
        order = _np.argsort(dev)
        mad = float(dev[order][_np.searchsorted(_np.cumsum(counts[order])/max(n,1),0.5)])
        mean = s/max(n,1)
        nz = _np.flatnonzero(counts)
        stats = dict(n=n, mean=mean, std=_math.sqrt(max(s2/max(n,1) - mean**2,0.0)), median=median, sigma=1.4826*mad,
                     p99=float(_np.interp(0.99,cdf,centres)), p999=float(_np.interp(0.999,cdf,centres)),
                     max=float(edges[nz[-1]+1]) if len(nz) else float('nan'))
        return counts, edges, stats

    def threshold(self, field='z', tomo_id=None, k=5.0, source='auto'):
        """Robust threshold of a tomogram: median + k sigma (sigma = 1.4826 MAD) of :meth:`histogram`.

        source ``'auto'``: the :meth:`run_null` null when loaded, else the whole slab (``'blocks'``)."""
        tid, m = self._manifest(tomo_id)
        if source == 'auto':
            source = 'null' if tid in self.null_manifests else 'blocks'
        _,_,st = self.histogram(field,tid,source=source)
        return st['median'] + k*st['sigma']

    def assemble(self, field='z', tomo_id=None, out=None):
        """Tomogram-sized volume (output grid of the manifest) of ``'z'``, ``'cc'``, ``'ang'`` or ``'blk'``.

        Where blocks overlap, the sample with the highest z (with its cc, angle
        and block); NaN (z, cc) or -1 (ang, blk) where no block searched.
        ``'blk'`` exists only for blocks with their own ali_eu.  ``out``: also
        write it (MRC).  Returns the volume, shape (Z, Y, X)."""
        tid, m = self._manifest(tomo_id)
        self._need_volumes(m)
        vol = self._read_vol(m,field)
        if out is not None:
            d = _os.path.dirname(out)
            if d and not _os.path.isdir(d):
                _os.makedirs(d)
            _mrc.write(vol,out,apix=m['tomogram']['apix_out'])
        return vol

    def coverage(self, tomo_id=None, slab=None, thickness=None):
        """Fraction of the tomogram searched: XY footprint and XYZ volume.

        XYZ is measured against the slab (``slab`` / ``thickness``, default
        :attr:`slab` / :attr:`thickness`, in angstroms / pixels as there), or
        the whole tomogram without them."""
        tid, m = self._manifest(tomo_id)
        b = int(m['tomogram']['bin'])
        apix = float(m['tomogram']['apix'])
        sx, sy, sz = m['tomogram']['size_out_xyz']
        self._need_volumes(m)
        cov = _np.isfinite(self._read_vol(m,'z'))
        slab = self._slab_of(tid) if slab is None else slab
        thickness = self.thickness if thickness is None else thickness
        size = _np.array(m['tomogram']['size_xyz'],dtype=float)
        if slab is None and thickness is None:
            region = _np.ones_like(cov)
        else:
            zc = _np.arange(sz)*b + (b-1)/2.0
            X, Y = _np.meshgrid(_np.arange(sx)*b + (b-1)/2.0,_np.arange(sy)*b + (b-1)/2.0,indexing='xy')
            z0 = self._slab_z0(slab,X,Y,size,apix)
            half = (size[2] if thickness is None else thickness)/2.0
            region = _np.abs(zc[:,None,None] - z0[None]) <= half
        xy = cov.any(axis=0)
        return dict(xy=float(xy.mean()), xyz=float((cov & region).sum()/max(region.sum(),1)),
                    region_voxels=int(region.sum()), searched_voxels=int(cov.sum()))

    def pick(self, field='z', tomo_id=None, threshold='auto', k=5.0, radius=None, max_peaks=None,
             mask=None, edge=0, refine=False, volume=None):
        """Peaks of a tomogram: local maxima above a threshold, at least ``radius`` apart.

        Parameters
        ----------
        field : str
            ``'z'`` (default) or ``'cc'``.
        threshold : float or ``'auto'``
            ``'auto'``: :meth:`threshold` with ``k``.
        radius : float
            Minimum distance between picks, in output voxels.  Default: 0.7 x
            the particle radius of the reference mask (needs :attr:`reference`),
            otherwise 4.  The mask radius itself is too large for packed
            particles: neighbouring ribosomes sit ~10 voxels apart at bin8 for a
            mask radius of 11.5, and a full-radius exclusion drops a quarter of
            them.
        max_peaks : int, optional
            Keep the best ones.
        mask : ndarray of bool, optional
            Output-grid mask of the allowed positions.
        edge : int
            Minimum distance to the tomogram border, in output voxels.
        refine : bool
            Sub-voxel position from a parabola through each peak and its
            neighbours.
        volume : ndarray, optional
            The assembled ``field`` volume, if already available.

        Returns
        -------
        dict
            tomo_id, voxel (output grid, x y z), position (tomogram voxels,
            x y z), score, z, cc, ang, block (ptcl_id), eu (SUSAN ZYZ, rad).
        """
        tid, m = self._manifest(tomo_id)
        b = int(m['tomogram']['bin'])
        vol = self.assemble(field,tid) if volume is None else volume
        thr = self.threshold(field,tid,k) if threshold == 'auto' else float(threshold)
        if radius is None:
            radius = 0.7*self.particle_radius()/b if self.reference is not None else 4.0
        v = _np.where(_np.isfinite(vol),vol,-_np.inf).astype(_np.float32)
        zyx = _ssa_utils.local_maxima(v,thr)
        if mask is not None:
            zyx = zyx[_np.asarray(mask,dtype=bool)[tuple(zyx.T)]]
        if edge > 0:
            e = int(edge)
            zyx = zyx[_np.all((zyx >= e) & (zyx < _np.array(v.shape) - e),axis=1)]
        score = v[tuple(zyx.T)]
        order = _np.argsort(-score,kind='stable')
        zyx, score = zyx[order], score[order]
        keep = _ssa_utils.suppress_peaks(zyx,v.shape,radius,max_peaks)
        zyx, score = zyx[keep], score[keep]

        sub = _np.zeros(zyx.shape)
        if refine:
            for ax in range(3):
                for i,p in enumerate(zyx):
                    if 0 < p[ax] < v.shape[ax]-1:
                        q0, q1 = p.copy(), p.copy()
                        q0[ax] -= 1
                        q1[ax] += 1
                        cm, c0, cp = v[tuple(q0)], v[tuple(p)], v[tuple(q1)]
                        den = cm - 2*c0 + cp
                        if _np.isfinite(den) and den < 0:
                            sub[i,ax] = _np.clip(0.5*(cm - cp)/den,-0.5,0.5)

        res = self._resolve_poses(m,zyx)
        voxel = (zyx + sub)[:,::-1]
        return dict(tomo_id=_np.full(len(zyx),tid,dtype=_np.uint32), voxel=voxel.astype(_np.float32),
                    position=(voxel*b + (b-1)/2.0).astype(_np.float32), score=score.astype(_np.float32), **res)

    def _resolve_poses(self, m, zyx):
        """Per pick: z, cc, ang and the pose R(ali_eu of the block) R(table[ang]) at that voxel."""
        n = len(zyx)
        z  = _np.full(n,-_np.inf,dtype=_np.float32)
        cc = _np.full(n,_np.nan,dtype=_np.float32)
        ang = _np.full(n,-1,dtype=_np.int64)
        bix = _np.full(n,-1,dtype=_np.int64)
        eu = _np.zeros((n,3),dtype=_np.float32)
        if n:
            loc = tuple(_np.asarray(zyx,dtype=int).T)
            z   = self._read_vol(m,'z')[loc].astype(_np.float32)
            cc  = self._read_vol(m,'cc')[loc].astype(_np.float32)
            ang = self._read_vol(m,'ang')[loc].astype(_np.int64)
            if 'blk' in m['volumes']:
                bix = self._read_vol(m,'blk')[loc].astype(_np.int64)
        ids = _np.array([blk['ptcl_id'] for blk in m['blocks']],dtype=_np.int64)
        blk_id = _np.where(bix >= 0,ids[_np.maximum(bix,0)] if len(ids) else -1,-1)
        ok = ang >= 0
        if ok.any():
            R = _ssa_utils.euZYZ_to_rotm(m['_table'][ang[ok]])
            if (bix[ok] >= 0).any():                       # blocks with their own ali_eu (no blk volume: all 0)
                ali = _np.array([blk['ali_eu'] for blk in m['blocks']],dtype=_np.float32)
                Ra = _ssa_utils.euZYZ_to_rotm(_np.where((bix[ok] >= 0)[:,None],ali[_np.maximum(bix[ok],0)],0))
                R = Ra @ R
            eu[ok] = _ssa_utils.rotm_to_euZYZ(R)
        return dict(z=z, cc=cc, ang=ang, block=blk_id, eu=eu)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def _tomograms_for(self, picks):
        if self._tomos is not None:
            return self._tomos
        m = self.manifests[int(picks['tomo_id'][0])]
        return _ssa_data.Tomograms(m['params']['tomos_file'])

    def to_particles(self, picks, filename=None):
        """Picks as :class:`susan.data.Particles` (pose in ali_eu, score in ali_cc); saved if filename."""
        T = self._tomograms_for(picks)
        P = _ssa_data.Particles.import_data(position=picks['position'],tomograms=T,tomos_id=picks['tomo_id'])
        order = P.ptcl_id.astype(_np.int64)
        for t in _np.unique(P.tomo_id):
            P.position[P.tomo_id == t] += T.tomo_position[int(T.get_cix(t))]
        if _np.any(T.tomo_position):
            P.update_defocus(T)
        P.ali_eu[0][:] = picks['eu'][order]
        P.ali_cc[0][:] = picks['score'][order]
        if filename is not None:
            P.save(filename)
        return P

    def to_artiax(self, picks, filename, tomo_id=None, scale_to_tomo=False):
        """Picks as an ArtiaX ``.tsv`` (:meth:`susan.data.Particles.export_artiax`): positions from the
        tomogram corner, in angstroms (default), or in voxels with ``scale_to_tomo=True``."""
        T = self._tomograms_for(picks)
        self.to_particles(picks).export_artiax(T,filename,tomo_id=tomo_id,scale_to_tomo=scale_to_tomo)

    def to_csv(self, picks, filename):
        """Picks as CSV: tomo_id, position (tomogram voxels), score, z, cc, ang, block, eu (rad)."""
        with open(filename,'w') as fp:
            fp.write('tomo_id,x,y,z,score,z_score,cc,ang,block,eu1,eu2,eu3\n')
            for i in range(len(picks['score'])):
                x,y,z = picks['position'][i]
                fp.write('%d,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f\n' % (
                    picks['tomo_id'][i],x,y,z,picks['score'][i],picks['z'][i],picks['cc'][i],picks['ang'][i],
                    picks['block'][i],*picks['eu'][i]))
