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

import susan.utils.txt_parser as _prsr
from susan.utils import is_extension as _is_ext
from susan.utils import force_extension as _force_ext

class Reference:
    def __init__(self,filename=None,n_refs=0):
        self.ref = []
        self.msk = []
        self.h1  = []
        self.h2  = []
        
        if isinstance(filename,str):
            fp = open(filename,"rb")
            num_refs = int(_prsr.read(fp,'num_ref'))
            for i in range(num_refs):
                self.ref.append(_prsr.read(fp,'map'))
                self.msk.append(_prsr.read(fp,'mask'))
                self.h1.append(_prsr.read(fp,'h1'))
                self.h2.append(_prsr.read(fp,'h2'))
        elif( n_refs>0 ):
            for i in range(n_refs):
                self.ref.append('')
                self.msk.append('')
                self.h1.append('')
                self.h2.append('')
    
    def get_n_refs(self): return len(self.ref)
    
    n_refs = property(get_n_refs)
    
    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'refstxt'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'refstxt') + '?')
    
    #def __repr__(self):
    #    return "Reference"
    
    @staticmethod
    def load(filename):
        Reference._check_filename(filename)
        result = Reference(filename=filename)
        return result
    
    def save(self,filename):
        Reference._check_filename(filename)
        
        if len(self.ref) is not len(self.msk):
            raise NameError('Reference entries do not match Mask entries')
            
        if len(self.ref) is not len(self.h1):
            raise NameError('Reference entries do not match Half1 entries')

        if len(self.ref) is not len(self.h2):
            raise NameError('Reference entries do not match Half2 entries')
        
        fp=open(filename,'w')
        _prsr.write(fp,'num_ref',str(self.n_refs))
        for i in range(self.n_refs):
            fp.write('## Reference '+str(i+1)+'\n')
            _prsr.write(fp,'map' ,self.ref[i])
            _prsr.write(fp,'mask',self.msk[i])
            _prsr.write(fp,'h1'  ,self.h1[i] )
            _prsr.write(fp,'h2'  ,self.h2[i] )
        fp.close()