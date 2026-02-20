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
from numba import jit as _jit
from susan.data  import Tomograms       as _tomodef
from susan.utils import is_extension    as _is_ext
from susan.utils import force_extension as _force_ext
from susan.utils import euZYZ_rotm      as _euZYZ_rotm
from ._PtclsGeom import PtclsGeom       as _Geom
from ._PtclsMRA  import PtclsMRA        as _MRA

class Particles:
    
    Geom = _Geom
    MRA  = _MRA
    
    def __init__(self,filename=None,n_ptcl=0,n_proj=0,n_refs=0):
        if isinstance(filename, str):
            self._load(filename) 
        else:
            if n_ptcl > 0 and n_proj > 0 and n_refs>0:
                self._alloc(n_ptcl,n_proj,n_refs)
            else:
                raise NameError('Invalid input')

    def get_n_ptcl(self): return self.ptcl_id.shape[0]
    def get_n_refs(self): return self.ali_eu.shape[0]
    def get_n_proj(self): return self.prj_eu.shape[1]
    
    n_ptcl = property(get_n_ptcl)
    n_refs = property(get_n_refs)
    n_proj = property(get_n_proj)
    
    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'ptclsraw'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'ptclsraw') + '?')
    
    def _alloc(self,n_ptcl,n_proj,n_refs):
        self.ptcl_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_cix = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.position = _np.zeros((n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ref_cix  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.half_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.extra_1  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        self.extra_2  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        
        # 3D alignment
        self.ali_eu   = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Radians
        self.ali_t    = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ali_cc   = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        self.ali_w    = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        
        # 2D alignment
        self.prj_eu   = _np.zeros((n_ptcl,n_proj,3),dtype=_np.float32) # in Radians
        self.prj_t    = _np.zeros((n_ptcl,n_proj,2),dtype=_np.float32) # in Angstroms
        self.prj_cc   = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        self.prj_w    = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # score

    def _load_header(self,fp):
        buffer = fp.read( 8 + 4*3 )
        if not _np.array_equal( buffer[:8], b'SsaPtcl1' ):
            raise NameError("Invalid File signature")
        return _np.frombuffer(buffer[8:],_np.uint32)
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _parse_buffer_stg1(ix,buffer,pid,tid,tcix,pos,rcix,hid,e1,e2):
        pid[ix],tid[ix],tcix[ix] = buffer[:3].view(_np.uint32)
        pos[ix] = buffer[3:6]
        rcix[ix],hid[ix] = buffer[6:8].view(_np.uint32)
        e1[ix],e2[ix] = buffer[8:10]
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _parse_buffer_stg2(ix,R,buffer,eu,t,cc,w):
        i = 0
        for j in range(R):
            eu[j,ix,0] = buffer[i  ]
            eu[j,ix,1] = buffer[i+1]
            eu[j,ix,2] = buffer[i+2]
            i = i+3
        
        for j in range(R):
            t[j,ix,0] = buffer[i  ]
            t[j,ix,1] = buffer[i+1]
            t[j,ix,2] = buffer[i+2]
            i = i+3
        
        for j in range(R):
            cc[j,ix] = buffer[i]
            i = i+1
        
        for j in range(R):
            w[j,ix] = buffer[i]
            i = i+1
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _parse_buffer_stg3(ix,P,buffer,eu,t,cc,w):
        i = 0
        for j in range(P):
            eu[ix,j,0] = buffer[i  ]
            eu[ix,j,1] = buffer[i+1]
            eu[ix,j,2] = buffer[i+2]
            i = i+3
        
        for j in range(P):
            t[ix,j,0] = buffer[i  ]
            t[ix,j,1] = buffer[i+1]
            i = i+2
        
        for j in range(P):
            cc[ix,j] = buffer[i]
            i = i+1
        
        for j in range(P):
            w[ix,j] = buffer[i]
            i = i+1
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _parse_buffer_stg4(ix,P,buffer,dU,dV,dA,ph,Bf,EF,res,scr):
        i = 0
        for j in range(P):
            dU [ix,j] = buffer[i  ]
            dV [ix,j] = buffer[i+1]
            dA [ix,j] = buffer[i+2]
            ph [ix,j] = buffer[i+3]
            Bf [ix,j] = buffer[i+4]
            EF [ix,j] = buffer[i+5]
            res[ix,j] = buffer[i+6]
            scr[ix,j] = buffer[i+7]
            i = i+8
    
    def _parse_buffer(self,ix,buffer):
        
        # Parse with NUMBA: 1.7x faster 
        Particles._parse_buffer_stg1(ix,buffer,
                                     self.ptcl_id,self.tomo_id,self.tomo_cix,
                                     self.position,self.ref_cix,self.half_id,
                                     self.extra_1,self.extra_2)
        
        # 3D alignment
        R = self.ali_eu.shape[0]
        Particles._parse_buffer_stg2(ix,R,buffer[10:],
                                     self.ali_eu,self.ali_t,
                                     self.ali_cc,self.ali_w)
        
        # 2D alignment
        P = self.prj_eu.shape[1]
        Particles._parse_buffer_stg3(ix,P,buffer[(10+8*R):],
                                     self.prj_eu,self.prj_t,
                                     self.prj_cc,self.prj_w)
        
        # Defocus
        Particles._parse_buffer_stg4(ix,P,buffer[(10+8*R+7*P):],
                                     self.def_U   ,self.def_V   ,self.def_ang,
                                     self.def_phas,self.def_Bfct,self.def_ExFl,
                                     self.def_mres,self.def_scor)
        
    def sort(self):
        idx = _np.lexsort((self.ptcl_id,self.tomo_id))
        self.ptcl_id  = self.ptcl_id [idx]
        self.tomo_id  = self.tomo_id [idx]
        self.tomo_cix = self.tomo_cix[idx]
        self.position = self.position[idx,:]
        self.ref_cix  = self.ref_cix [idx]
        self.half_id  = self.half_id [idx]
        self.extra_1  = self.extra_1 [idx]
        self.extra_2  = self.extra_2 [idx]
        # 3D alignment
        self.ali_eu   = self.ali_eu[:,idx,:]
        self.ali_t    = self.ali_t [:,idx,:]
        self.ali_cc   = self.ali_cc[:,idx]
        self.ali_w    = self.ali_w [:,idx]
        # 2D alignment
        self.prj_eu   = self.prj_eu[idx,:,:]
        self.prj_t    = self.prj_t [idx,:,:]
        self.prj_cc   = self.prj_cc[idx,:]
        self.prj_w    = self.prj_w [idx,:]
        # Defocus
        self.def_U    = self.def_U   [idx,:]
        self.def_V    = self.def_V   [idx,:]
        self.def_ang  = self.def_ang [idx,:]
        self.def_phas = self.def_phas[idx,:]
        self.def_Bfct = self.def_Bfct[idx,:]
        self.def_ExFl = self.def_ExFl[idx,:]
        self.def_mres = self.def_mres[idx,:]
        self.def_scor = self.def_scor[idx,:]

    def _load(self,filename):
        Particles._check_filename(filename)
        
        fp = open(filename,"rb")
        n_ptcl, n_proj, n_refs = self._load_header(fp)
        self._alloc(n_ptcl,n_proj,n_refs)
        bytes_per_ptcl = 4*( 10 + 8*n_refs + 7*n_proj + 8*n_proj )
        for i in range(n_ptcl):
            buffer = _np.frombuffer(fp.read(bytes_per_ptcl),_np.float32)
            self._parse_buffer(i,buffer)
        fp.close()
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _set_buffer_stg1(ix,buffer,pid,tid,tcix,pos,rcix,hid,e1,e2):
        tmp = _np.uint32(pid[ix])
        buffer[0] = tmp.view(_np.float32)
        tmp = _np.uint32(tid[ix])
        buffer[1] = tmp.view(_np.float32)
        tmp = _np.uint32(tcix[ix])
        buffer[2] = tmp.view(_np.float32)
        buffer[3:6] = pos[ix]
        tmp = _np.uint32(rcix[ix])
        buffer[6] = tmp.view(_np.float32)
        tmp = _np.uint32(hid[ix])
        buffer[7] = tmp.view(_np.float32)
        buffer[8] = e1[ix]
        buffer[9] = e2[ix]
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _set_buffer_stg2(ix,R,buffer,eu,t,cc,w):
        i = 0
        for j in range(R):
            buffer[i  ] = eu[j,ix,0]
            buffer[i+1] = eu[j,ix,1]
            buffer[i+2] = eu[j,ix,2]
            i = i+3
        
        for j in range(R):
            buffer[i  ] = t[j,ix,0]
            buffer[i+1] = t[j,ix,1]
            buffer[i+2] = t[j,ix,2]
            i = i+3
        
        for j in range(R):
            buffer[i] = cc[j,ix]
            i = i+1
        
        for j in range(R):
            buffer[i] = w[j,ix]
            i = i+1
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _set_buffer_stg3(ix,P,buffer,eu,t,cc,w):
        i = 0
        for j in range(P):
            buffer[i  ] = eu[ix,j,0]
            buffer[i+1] = eu[ix,j,1]
            buffer[i+2] = eu[ix,j,2]
            i = i+3
        
        for j in range(P):
            buffer[i  ] = t[ix,j,0]
            buffer[i+1] = t[ix,j,1]
            i = i+2
        
        for j in range(P):
            buffer[i] = cc[ix,j]
            i = i+1
        
        for j in range(P):
            buffer[i] = w[ix,j]
            i = i+1
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _set_buffer_stg4(ix,P,buffer,dU,dV,dA,ph,Bf,EF,res,scr):
        i = 0
        for j in range(P):
            buffer[i  ] = dU [ix,j]
            buffer[i+1] = dV [ix,j]
            buffer[i+2] = dA [ix,j]
            buffer[i+3] = ph [ix,j]
            buffer[i+4] = Bf [ix,j]
            buffer[i+5] = EF [ix,j]
            buffer[i+6] = res[ix,j]
            buffer[i+7] = scr[ix,j]
            i = i+8
    
    def save(self,filename):
        Particles._check_filename(filename)
        
        fp = open(filename,"wb")
        fp.write( b'SsaPtcl1' )
        _np.array( (self.n_ptcl,self.n_proj,self.n_refs), dtype=_np.uint32 ).tofile(fp)
        
        R = self.ali_eu.shape[0]
        P = self.prj_eu.shape[1]        
        buffer = _np.zeros(10+8*R+7*P+8*P,dtype=_np.float32)
        
        for ix in range(self.n_ptcl):
            
            Particles._set_buffer_stg1(ix,buffer,
                                         self.ptcl_id,self.tomo_id,self.tomo_cix,
                                         self.position,self.ref_cix,self.half_id,
                                         self.extra_1,self.extra_2)
            
            # 3D alignment
            Particles._set_buffer_stg2(ix,R,buffer[10:],
                                         self.ali_eu,self.ali_t,
                                         self.ali_cc,self.ali_w)
            
            # 2D alignment
            Particles._set_buffer_stg3(ix,P,buffer[(10+8*R):],
                                         self.prj_eu,self.prj_t,
                                         self.prj_cc,self.prj_w)
            
            # Defocus
            Particles._set_buffer_stg4(ix,P,buffer[(10+8*R+7*P):],
                                         self.def_U   ,self.def_V   ,self.def_ang,
                                         self.def_phas,self.def_Bfct,self.def_ExFl,
                                         self.def_mres,self.def_scor)
            buffer.tofile(fp)
        fp.close()
    
    def __getitem__(self,idx):
        if isinstance(idx,slice):
            idx = _np.arange(idx.start,idx.stop,idx.step)
        return self.select(idx)
    
    def select(self,idx):
        idx = _np.array(idx)
        if( idx.ndim >= 2 ):
            idx = idx[:,0]
            number_of_particles = idx.shape[0]
        elif( idx.ndim == 1):
            number_of_particles = idx.shape[0]
        elif( idx.ndim == 0):
            number_of_particles = 1
        ptcls_out = Particles(n_ptcl=number_of_particles,n_proj=self.n_proj,n_refs=self.n_refs)
        if (number_of_particles > 1):
            ptcls_out.ptcl_id  = self.ptcl_id [idx]
            ptcls_out.tomo_id  = self.tomo_id [idx]
            ptcls_out.tomo_cix = self.tomo_cix[idx]
            ptcls_out.position = self.position[idx,:]
            ptcls_out.ref_cix  = self.ref_cix [idx]
            ptcls_out.half_id  = self.half_id [idx]
            ptcls_out.extra_1  = self.extra_1 [idx]
            ptcls_out.extra_2  = self.extra_2 [idx]
            # 3D alignment
            ptcls_out.ali_eu   = self.ali_eu[:,idx,:]
            ptcls_out.ali_t    = self.ali_t [:,idx,:]
            ptcls_out.ali_cc   = self.ali_cc[:,idx]
            ptcls_out.ali_w    = self.ali_w [:,idx]
            # 2D alignment
            ptcls_out.prj_eu   = self.prj_eu[idx,:,:]
            ptcls_out.prj_t    = self.prj_t [idx,:,:]
            ptcls_out.prj_cc   = self.prj_cc[idx,:]
            ptcls_out.prj_w    = self.prj_w [idx,:]
            # Defocus
            ptcls_out.def_U    = self.def_U   [idx,:]
            ptcls_out.def_V    = self.def_V   [idx,:]
            ptcls_out.def_ang  = self.def_ang [idx,:]
            ptcls_out.def_phas = self.def_phas[idx,:]
            ptcls_out.def_Bfct = self.def_Bfct[idx,:]
            ptcls_out.def_ExFl = self.def_ExFl[idx,:]
            ptcls_out.def_mres = self.def_mres[idx,:]
            ptcls_out.def_scor = self.def_scor[idx,:]
        elif (number_of_particles == 1):
            ptcls_out.ptcl_id  = _np.expand_dims(_np.array(self.ptcl_id [idx]), 0).astype('uint32')
            ptcls_out.tomo_id  = _np.expand_dims(_np.array(self.tomo_id [idx]), 0).astype('uint32')
            ptcls_out.tomo_cix = _np.expand_dims(_np.array(self.tomo_cix[idx]), 0).astype('uint32')
            ptcls_out.position = _np.expand_dims(_np.array(self.position[idx,:]), 0).astype('float32')
            ptcls_out.ref_cix  = _np.expand_dims(_np.array(self.ref_cix [idx]), 0).astype('uint32')
            ptcls_out.half_id  = _np.expand_dims(_np.array(self.half_id [idx]), 0).astype('uint32')
            ptcls_out.extra_1  = _np.expand_dims(_np.array(self.extra_1 [idx]), 0).astype('float32')
            ptcls_out.extra_2  = _np.expand_dims(_np.array(self.extra_2 [idx]), 0).astype('float32')
            # 3D alignment
            ptcls_out.ali_eu   = _np.expand_dims(_np.array(self.ali_eu[:,idx,:]),1).astype('float32')
            ptcls_out.ali_t    = _np.expand_dims(_np.array(self.ali_t [:,idx,:]),1).astype('float32')
            ptcls_out.ali_cc   = _np.expand_dims(_np.array(self.ali_cc[:,idx])  ,1).astype('float32')
            ptcls_out.ali_w    = _np.expand_dims(_np.array(self.ali_w [:,idx])  ,1).astype('float32')
            # 2D alignment
            ptcls_out.prj_eu   = _np.expand_dims(_np.array(self.prj_eu[idx,:,:]),0).astype('float32')
            ptcls_out.prj_t    = _np.expand_dims(_np.array(self.prj_t [idx,:,:]),0).astype('float32')
            ptcls_out.prj_cc   = _np.expand_dims(_np.array(self.prj_cc[idx,:]),0).astype('float32')
            ptcls_out.prj_w    = _np.expand_dims(_np.array(self.prj_w [idx,:]),0).astype('float32')
            # Defocus
            ptcls_out.def_U    = _np.expand_dims(_np.array(self.def_U   [idx,:]),0).astype('float32')
            ptcls_out.def_V    = _np.expand_dims(_np.array(self.def_V   [idx,:]),0).astype('float32')
            ptcls_out.def_ang  = _np.expand_dims(_np.array(self.def_ang [idx,:]),0).astype('float32')
            ptcls_out.def_phas = _np.expand_dims(_np.array(self.def_phas[idx,:]),0).astype('float32')
            ptcls_out.def_Bfct = _np.expand_dims(_np.array(self.def_Bfct[idx,:]),0).astype('float32')
            ptcls_out.def_ExFl = _np.expand_dims(_np.array(self.def_ExFl[idx,:]),0).astype('float32')
            ptcls_out.def_mres = _np.expand_dims(_np.array(self.def_mres[idx,:]),0).astype('float32')
            ptcls_out.def_scor = _np.expand_dims(_np.array(self.def_scor[idx,:]),0).astype('float32')
        # Sort
        if (number_of_particles > 1):
            ptcls_out.sort()
        return ptcls_out
    
    def append_ptcls(self,ptcls):
        self.ptcl_id  = _np.concatenate( (self.ptcl_id ,ptcls.ptcl_id ),axis=0 )
        self.tomo_id  = _np.concatenate( (self.tomo_id ,ptcls.tomo_id ),axis=0 )
        self.tomo_cix = _np.concatenate( (self.tomo_cix,ptcls.tomo_cix),axis=0 )
        self.position = _np.concatenate( (self.position,ptcls.position),axis=0 )
        self.ref_cix  = _np.concatenate( (self.ref_cix ,ptcls.ref_cix ),axis=0 )
        self.half_id  = _np.concatenate( (self.half_id ,ptcls.half_id ),axis=0 )
        self.extra_1  = _np.concatenate( (self.extra_1 ,ptcls.extra_1 ),axis=0 )
        self.extra_2  = _np.concatenate( (self.extra_2 ,ptcls.extra_2 ),axis=0 )
        # 3D alignment
        self.ali_eu   = _np.concatenate( (self.ali_eu,ptcls.ali_eu),axis=1 )
        self.ali_t    = _np.concatenate( (self.ali_t ,ptcls.ali_t ),axis=1 )
        self.ali_cc   = _np.concatenate( (self.ali_cc,ptcls.ali_cc),axis=1 )
        self.ali_w    = _np.concatenate( (self.ali_w ,ptcls.ali_w ),axis=1 )
        # 2D alignment
        self.prj_eu   = _np.concatenate( (self.prj_eu,ptcls.prj_eu),axis=0 )
        self.prj_t    = _np.concatenate( (self.prj_t ,ptcls.prj_t ),axis=0 )
        self.prj_cc   = _np.concatenate( (self.prj_cc,ptcls.prj_cc),axis=0 )
        self.prj_w    = _np.concatenate( (self.prj_w ,ptcls.prj_w ),axis=0 )
        # Defocus
        self.def_U    = _np.concatenate( (self.def_U   ,ptcls.def_U   ),axis=0 )
        self.def_V    = _np.concatenate( (self.def_V   ,ptcls.def_V   ),axis=0 )
        self.def_ang  = _np.concatenate( (self.def_ang ,ptcls.def_ang ),axis=0 )
        self.def_phas = _np.concatenate( (self.def_phas,ptcls.def_phas),axis=0 )
        self.def_Bfct = _np.concatenate( (self.def_Bfct,ptcls.def_Bfct),axis=0 )
        self.def_ExFl = _np.concatenate( (self.def_ExFl,ptcls.def_ExFl),axis=0 )
        self.def_mres = _np.concatenate( (self.def_mres,ptcls.def_mres),axis=0 )
        self.def_scor = _np.concatenate( (self.def_scor,ptcls.def_scor),axis=0 )
        # Sort
        self.sort()

    def set_weights(self,in_wgt):
        mask = (self.prj_w > 0).transpose()
        mask = mask * in_wgt
        self.prj_w[:,:] = mask.transpose()
        
    def halfsets_by_Y(self):
        tomo_ids = _np.unique( self.tomo_id )
        for tid in tomo_ids:
            idx = self.tomo_id == tid
            self.half_id[idx] = 1
            th = _np.quantile( self.position[idx,1].flatten() ,0.5)
            self.half_id[idx] = self.half_id[idx] + (self.position[idx,1]>th)

    def halfsets_even_odd(self):
        self.half_id[0::2] = 1
        self.half_id[1::2] = 2

    def halfsets_randomize(self):
        self.half_id[:] = _np.random.randint(1,3,self.half_id.shape)
        
    def update_position(self,ref_id=0):
        self.position = self.position + self.ali_t[ref_id]
        self.ali_t[ref_id,:,:] = 0

    @staticmethod
    @_jit(nopython=True,cache=True)
    def _update_new_defocus(dU,dV,R,p,num_proj,z_coef,dU_in,dV_in):
        for i in range(num_proj):
            dZ = R[i,2,0]*p[0] + R[i,2,1]*p[1] + R[i,2,2]*p[2]
            dU[i] = dU_in[i] + z_coef*dZ
            dV[i] = dV_in[i] + z_coef*dZ

    def update_defocus(self,tomos_info,ref_id=0,z_sign=None):
        
        # Calculate tilt rotation matrix
        R_arr = _np.zeros((tomos_info.n_tomos,tomos_info.n_projs,3,3),dtype=_np.float32)
        for t in range(tomos_info.n_tomos):
            for p in range(tomos_info.num_proj[t]):
                _euZYZ_rotm(R_arr[t,p],_np.deg2rad(tomos_info.proj_eZYZ[t,p]))
        
        # Update defocus
        for k in range(self.n_ptcl):
            tid = self.tomo_cix[k]
            
            self.prj_w   [k,:] = tomos_info.proj_wgt[tid,:]
            self.def_ang [k,:] = tomos_info.def_ang [tid,:]
            self.def_phas[k,:] = tomos_info.def_phas[tid,:]
            self.def_Bfct[k,:] = tomos_info.def_Bfct[tid,:]
            self.def_ExFl[k,:] = tomos_info.def_ExFl[tid,:]
            self.def_mres[k,:] = tomos_info.def_mres[tid,:]
            self.def_scor[k,:] = tomos_info.def_scor[tid,:]
            
            # Note: Numba makes it ~23.3 times faster
            pos = self.position[k] + self.ali_t[ref_id,k]
            if z_sign is None:
                z_sign = tomos_info.handedness[tid]
            Particles._update_new_defocus(
                self.def_U[k],
                self.def_V[k],
                R_arr[tid],
                pos,
                tomos_info.num_proj[tid],
                z_sign,
                tomos_info.def_U[tid],
                tomos_info.def_V[tid]
            )

    def x(self,ref_idx=0):
        return self.position[:,0] + self.ali_t[ref_idx,:,0]

    def y(self,ref_idx=0):
        return self.position[:,1] + self.ali_t[ref_idx,:,1]

    def z(self,ref_idx=0):
        return self.position[:,2] + self.ali_t[ref_idx,:,2]

    def pos(self,ref_idx=0):
        return self.position + self.ali_t[ref_idx]

    @staticmethod
    def _get_tomo_limit_angstroms(tomo_size,tomo_apix,border):
        return tomo_apix*( tomo_size-border )/2

    @staticmethod
    def _validate_tomogram(tomograms):
        if not isinstance(tomograms,_tomodef.Tomograms):
            raise ValueError('Tomograms must be a Tomograms object.')
        apix = _np.unique( tomograms.pix_size )
        if apix.size != 1:
            raise ValueError('Tomograms must have the same pixel size.')
        return apix[0]

    @staticmethod
    def _get_grid_step(s_ang,s_pix,apix):
        if   s_ang is     None and s_pix is     None:
            raise ValueError('Set the steps either in angstroms or pixels')
        elif s_ang is not None and s_pix is not None:
            raise ValueError('Set step_angstroms or step_pixels, not both.')
        elif s_ang is not None and s_pix is     None:
            return s_ang
        else:
            return s_pix*apix

    @staticmethod
    def _get_border_pixels(pix):
        p = _np.array(pix,_np.int32)
        if p.size == 1:
            return _np.array((pix,pix,pix),_np.int32)
        elif p.size == 3:
            return p
        else:
            raise ValueError('skip_border_pixels must be either a scalar or a 3-element vector')

    @staticmethod
    def grid_2d(tomograms,step_angstroms=None,step_pixels=None,skip_border_pixels=0,angle_deg_Y=0):
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)

        R = _np.eye(3)
        _euZYZ_rotm(R,_np.deg2rad(_np.array((0,angle_deg_Y,0))))
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tcx = _np.zeros((0),dtype=_np.uint32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]) )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]) )
            x,y,z = _np.float32(_np.meshgrid(t_x,t_y,(0)))
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pos = pos@R
            pts = _np.concatenate( (pts,pos) )
            tcx = _np.concatenate( (tcx,_np.repeat(_np.uint32(i),pos.shape[0])) )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])) )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_cix[:]   = tcx
        ptcls.tomo_id [:]   = tid
        ptcls.ali_w[:] 	    = 1
        ptcls.update_defocus(tomograms)
        return ptcls
    
    @staticmethod
    def grid_3d(tomograms,step_angstroms=None,step_pixels=None,skip_border_pixels=0):
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tcx = _np.zeros((0),dtype=_np.uint32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]) )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]) )
            t_z = _np.arange(0,tomo_range[2],step,dtype=_np.float32)
            t_z = _np.concatenate( (-t_z[::-1],t_z[1:]) )
            x,y,z = _np.float32(_np.meshgrid(t_x,t_y,t_z))
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pts = _np.concatenate( (pts,pos) )
            tcx = _np.concatenate( (tcx,_np.repeat(_np.uint32(i),pos.shape[0])) )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])) )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_cix[:]   = tcx
        ptcls.tomo_id [:]   = tid
        ptcls.ali_w[:]      = 1
        ptcls.halfsets_even_odd()
        ptcls.update_defocus(tomograms)
        return ptcls

    @staticmethod
    def _validate_import_args(position,ptcls_id,tomos_id):
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError('Position must be a N-by-3 2D matrix')
        N = position.shape[0]
        if ptcls_id.shape[0] != N:
            raise ValueError('Number of entries in ptcls_id do not match position')
        if tomos_id.shape[0] != N:
            raise ValueError('Number of entries in tomos_id do not match position')
        return N
        
    @staticmethod
    def _calc_tomo_cix(tomo_cix,tomograms,tomos_id):
        LUT = {}
        for tid in range(tomograms.n_tomos):
            LUT[ int(tomograms.tomo_id[tid]) ] = tid
        for i in range(tomos_id.shape[0]):
            tomo_cix[i] = LUT[int(tomos_id[i])]
    
    @staticmethod
    def _calc_position(p_out,p_in,tomograms,tomos_cix,apix):
        for i in range(p_in.shape[0]):
            pos = p_in[i,:] - tomograms.tomo_size[tomos_cix[i]]/2
            p_out[i,:] = apix*pos
            
    @staticmethod
    def import_data(tomograms,position,tomos_id,ptcls_id=None,randomize_angles=False):
        if ptcls_id is None:
            ptcls_id = _np.arange(tomos_id.shape[0])
        apix = Particles._validate_tomogram(tomograms)
        N = Particles._validate_import_args(position,ptcls_id,tomos_id)
        ptcls = Particles(n_ptcl=N,n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = ptcls_id
        ptcls.tomo_id [:]   = tomos_id
        ptcls.ali_w[:]      = 1
        Particles._calc_tomo_cix(ptcls.tomo_cix,tomograms,tomos_id)
        Particles._calc_position(ptcls.position,position,tomograms,ptcls.tomo_cix,apix)
        ptcls.halfsets_even_odd()
        ptcls.sort()
        ptcls.update_defocus(tomograms)
        if randomize_angles:
            ptcls.ali_eu[:,:,:] = _np.random.uniform(0,_np.pi,ptcls.ali_eu.shape)
        return ptcls
    
    def export_positions(self,tomograms,ref_cix=0):
        apix = Particles._validate_tomogram(tomograms)
        pos = _np.zeros_like(self.pos(ref_cix))
        for i in range(pos.shape[0]):
            tmp = self.position[i,:] + self.ali_t[ref_cix,i,:]
            tmp = tmp/apix
            pos[i,:] = tmp + tomograms.tomo_size[self.tomo_cix[i]]/2
        return pos





