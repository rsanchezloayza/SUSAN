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

import numpy as _np

import susan.data    as _ssa_data
import susan.utils   as _ssa_utils
import susan.modules as _ssa_modules

from os.path import exists as _file_exists
from os.path import normpath   as _normpath
from os.path import commonpath as _commonpath
from os.path import relpath    as _relpath

class SubtomogramGenerator:
    """Generate subtomograms and export them with RELION-compatible metadata.

    .. warning::
        This is an experimental class.  The interface and output format may
        change in future versions.

    Reconstructs per-particle subtomograms via :class:`~susan.modules.SubtomoRec`
    and writes a RELION 3.1 STAR file containing the optics group and particle
    table.  Call one of the ``configure_*`` methods before
    :meth:`generate_subtomos` to set the reconstruction parameters for the
    target software.

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use for reconstruction.  Default: ``[0]``.
    subtomo_rec : :class:`~susan.modules.SubtomoRec`
        Underlying reconstruction module.  Advanced parameters can be
        adjusted directly on this object after calling a ``configure_*``
        method.
    """

    def __init__(self):
        self.list_gpus_ids = [0]
        self.subtomo_rec   = _ssa_modules.SubtomoRec()

    def _relion_relative_folder(self,metadata_file,subtomos_folder):
        md_f = _normpath(metadata_file)
        st_f = _normpath(subtomos_folder)
        base = _commonpath([md_f, st_f])
        return _relpath(st_f,base)

    def configure_relion(self, rec_ctf=True, invert_contrast=True):
        """Configure reconstruction settings for RELION.

        Sets normalisation, contrast, and CTF correction parameters on
        :attr:`subtomo_rec` to match RELION's conventions.

        Parameters
        ----------
        rec_ctf : bool, optional
            If ``True`` (default), use pre-Wiener CTF correction and write a
            per-particle CTF image (``_rlnCtfImage``).  If ``False``, use
            standard Wiener filter and write per-particle defocus columns.
        invert_contrast : bool, optional
            Invert the contrast of the output subtomograms.  Default: ``True``.
        """
        self.subtomo_rec.normalize_type   = 'zero_mean_one_std'
        self.subtomo_rec.normalize_output = True
        self.subtomo_rec.invert_contrast  = invert_contrast
        self.subtomo_rec.boost_lowfreq.scale = 0
        if rec_ctf:
            self.subtomo_rec.ctf_correction   = 'pre_wiener'
            self.subtomo_rec.relion_ctf       = True
        else:
            self.subtomo_rec.ctf_correction   = 'wiener'
            self.subtomo_rec.relion_ctf       = False

    def configure_dynamo(self, invert_contrast=False):
        """Configure reconstruction settings for DYNAMO.

        Sets normalisation, contrast, and CTF correction parameters on
        :attr:`subtomo_rec` to match DYNAMO's conventions (Wiener filter,
        no low-frequency boost).

        Parameters
        ----------
        invert_contrast : bool, optional
            Invert the contrast of the output subtomograms.  Default: ``False``.
        """
        self.subtomo_rec.normalize_type   = 'zero_mean_one_std'
        self.subtomo_rec.normalize_output = True
        self.subtomo_rec.invert_contrast  = invert_contrast
        self.subtomo_rec.ctf_correction   = 'wiener'
        self.subtomo_rec.boost_lowfreq.scale = 0


    def generate_subtomos(self, metadata_file, subtomos_folder, tomos_file, ptcls_in, box_size):
        """Reconstruct subtomograms and write a RELION STAR metadata file.

        The subtomogram MRC files are written to *subtomos_folder*.
        A RELION 3.1 STAR file is then written to *metadata_file* containing
        an ``data_optics`` block (single optics group derived from the first
        tomogram) and a ``data_particles`` block with one row per particle.
        Currently only ``.star`` output is supported.

        Parameters
        ----------
        metadata_file : str
            Output path for the STAR file (must end in ``.star``).
        subtomos_folder : str
            Directory where the per-particle MRC files will be written.
        tomos_file : str
            Path to the ``.tomostxt`` file describing the tomogram stack.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Side length of the cubic subtomogram box in pixels.
        """
        out_type = _ssa_utils.get_extension(metadata_file).lower()

        if subtomos_folder[-1] != '/':
            subtomos_folder += '/'

        if out_type in ['star',]:

            # Step 1: generate subtomograms
            self.subtomo_rec.list_gpus_ids    = self.list_gpus_ids
            self.subtomo_rec.bandpass.lowpass = box_size//2
            self.subtomo_rec.reconstruct(subtomos_folder,tomos_file,ptcls_in,box_size)

            # Step 2: generate metadata (star file)
            subtomos_path = self._relion_relative_folder(metadata_file, subtomos_folder)

            tomos = _ssa_data.Tomograms(tomos_file)
            ptcls = _ssa_data.Particles(ptcls_in)
            pos   = ptcls.export_positions(tomos)

            with open(metadata_file, 'w') as f:
                # Write optics group
                f.write('\n# version 30001\n\n')
                f.write('data_optics\n\nloop_\n')
                f.write('_rlnOpticsGroupName     #1\n')
                f.write('_rlnOpticsGroup         #2\n')
                f.write('_rlnImageSize           #3\n')
                f.write('_rlnImagePixelSize      #4\n')
                f.write('_rlnVoltage             #5\n')
                f.write('_rlnSphericalAberration #6\n')
                f.write('_rlnAmplitudeContrast   #7\n')
                f.write('_rlnImageDimensionality #8\n')
                f.write(f'opticsGroup1 1 {box_size} {tomos.pix_size[0]:.3f} {tomos.voltage [0]:.1f} {tomos.sph_aber[0]:.1f} {tomos.amp_cont[0]:.3f} 3 \n\n')

                # Write particles block
                f.write('\n# version 30001\n\n')
                f.write('data_particles\n\nloop_\n')
                f.write('_rlnImageName    #1\n')
                f.write('_rlnOpticsGroup  #2\n')

                f.write('_rlnAngleRot     #3\n')
                f.write('_rlnAngleTilt    #4\n')
                f.write('_rlnAnglePsi     #5\n')

                f.write('_rlnOriginX      #6\n')
                f.write('_rlnOriginY      #7\n')
                f.write('_rlnOriginZ      #8\n')

                f.write('_rlnCoordinateX  #9\n')
                f.write('_rlnCoordinateY  #10\n')
                f.write('_rlnCoordinateZ  #11\n')

                f.write('_rlnRandomSubset #12\n')

                if self.subtomo_rec.relion_ctf:
                    f.write('_rlnCtfImage         #13\n')
                else:
                    f.write('_rlnDefocusU         #13\n')
                    f.write('_rlnDefocusV         #14\n')
                    f.write('_rlnDefocusAngle     #15\n')
                    f.write('_rlnPhaseShift       #16\n')

                for idx in range(ptcls.n_ptcl):
                    p_id = ptcls.ptcl_id[idx]
                    file = f'{subtomos_path}/particle_{p_id:06d}.mrc'
                    fctf = f'{subtomos_path}/particle_{p_id:06d}.ctf.mrc'

                    ang_R = _np.rad2deg( -ptcls.ali_eu[0,idx,2] )
                    ang_T = _np.rad2deg( -ptcls.ali_eu[0,idx,1] )
                    ang_P = _np.rad2deg( -ptcls.ali_eu[0,idx,0] )

                    if _file_exists(f'{subtomos_folder}/particle_{p_id:06d}.mrc'):
                        #            1   2
                        f.write(f'{file} 1 ')

                        #              3           4           5       6   7   8
                        f.write(f'{ang_R:.2f} {ang_T:.2f} {ang_P:.2f} 0.0 0.0 0.0 ')

                        #                    9                10                11
                        f.write(f'{pos[idx,0]:4.0f} {pos[idx,1]:4.0f} {pos[idx,2]:4.0f} ')

                        #            12
                        f.write(f'{ptcls.half_id[idx]:1d} ')

                        if self.subtomo_rec.relion_ctf:
                            f.write(f'{fctf}\n')
                        else:
                            def_u = _np.median( ptcls.def_U   [idx] )
                            def_v = _np.median( ptcls.def_V   [idx] )
                            def_a = _np.median( ptcls.def_ang [idx] )
                            ph_sh = _np.median( ptcls.def_phas[idx] )
                            ph_sh = _np.rad2deg( ph_sh )
                            f.write(f'{def_u:5.0f} {def_v:5.0f} {def_a:5.1f} {ph_sh:5.1f}\n')

        else:
            print('Error: unsupported output type ' + out_type)

###########################################

class ProjectionExtractor:
    """Crop 2-D projection patches and export them with RELION-compatible metadata.

    .. warning::
        This is an experimental class.  The interface and output format may
        change in future versions.

    Uses :class:`~susan.modules.CropProjection` to extract per-particle
    2-D projection stacks and writes a RELION STAR file
    (``particles.star``) inside the output folder.

    .. rubric:: Attributes

    Attributes
    ----------
    proj_cropper : :class:`~susan.modules.CropProjection`
        Underlying cropping module.  Parameters (e.g. ``normalize_type``)
        can be adjusted directly on this object before calling
        :meth:`crop_projections`.
    """

    def __init__(self, num_threads=1):
        """
        Parameters
        ----------
        num_threads : int, optional
            Number of CPU threads for the cropping step.  Default: 1.
        """
        self.proj_cropper = _ssa_modules.CropProjection()
        self.proj_cropper.num_threads = num_threads
        self.proj_cropper.invert_contrast = True

    def crop_projections(self, output_folder, tomos_file, ptcls_in, box_size):
        """Crop projection patches and write a RELION STAR metadata file.

        Projection stacks are written by :attr:`proj_cropper` as
        ``stack_NN.txt`` files in *output_folder*.  These are then
        assembled into a single RELION 3.1 STAR file
        ``<output_folder>/particles.star`` with ``data_optics`` and
        ``data_particles`` blocks.

        Parameters
        ----------
        output_folder : str
            Directory where the cropped stacks and STAR file are written.
        tomos_file : str
            Path to the ``.tomostxt`` file describing the tomogram stack.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Side length of the 2-D crop box in pixels.
        """

        self.proj_cropper.reconstruct(output_folder,tomos_file,ptcls_in,box_size)

        tomos = _ssa_data.Tomograms(tomos_file)
        pixel_size = tomos.pix_size[0]
        voltage    = tomos.voltage [0]
        cs         = tomos.sph_aber[0]
        ac         = tomos.amp_cont[0]

        with open(f'{output_folder}/particles.star','w') as f:
            f.write('\ndata_optics\n\nloop_\n')
            f.write('_rlnOpticsGroupName\n')
            f.write('_rlnOpticsGroup\n')
            f.write('_rlnImageSize\n')
            f.write('_rlnImagePixelSize\n')
            f.write('_rlnVoltage\n')
            f.write('_rlnSphericalAberration\n')
            f.write('_rlnAmplitudeContrast\n')
            f.write('_rlnImageDimensionality\n')
            f.write(f'opticsGroup1 1 {box_size} {pixel_size:.3f} {voltage:.1f} {cs:.1f} {ac:.3f} 2\n\n')

            # Write particles block
            f.write('data_particles\n\nloop_\n')
            f.write('_rlnImageName        #1\n')
            f.write('_rlnCoordinateX      #2\n')
            f.write('_rlnCoordinateY      #3\n')
            f.write('_rlnCoordinateZ      #4\n')
            f.write('_rlnMicrographName   #5\n')
            f.write('_rlnDefocusU         #6\n')
            f.write('_rlnDefocusV         #7\n')
            f.write('_rlnDefocusAngle     #8\n')
            f.write('_rlnPhaseShift       #9\n')
            f.write('_rlnAngleRot        #10\n')
            f.write('_rlnAngleTilt       #11\n')
            f.write('_rlnAnglePsi        #12\n')
            f.write('_rlnOriginX         #13\n')
            f.write('_rlnOriginY         #14\n')
            f.write('_rlnRandomSubset    #15\n')
            f.write('_rlnOpticsGroup     #16\n')

            for i in range(self.proj_cropper.num_threads):
                with open(f'{output_folder}/stack_{i:02d}.txt','r') as f_in:
                    for line in f_in:
                        f.write(line.strip() + '  1\n')


