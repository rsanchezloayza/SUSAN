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

__all__ = ['read_line','read','parse_args','write']

def _decode_if_needed(line):
    try:
        return line.decode('utf-8')
    except:
        return line

def read_line(fp):
    line = _decode_if_needed( fp.readline().strip() )
    while len(line) > 0 and line[0] == "#" :
        line = _decode_if_needed( fp.readline().strip() )
    return line

def parse_args(fp):
    """Read one block of ``key: value`` lines into a dictionary.

    Mirrors the C++ ``IO::TxtParser::parse_args``: leading comment lines are
    skipped, key/value pairs are collected, and the block ends at the first
    comment line that follows a pair (or at a line that is not a pair, or at
    the end of the file).  Whitespace around both the key and the value is
    removed, so ``key:value`` and ``key: value`` are equivalent.

    Parameters
    ----------
    fp : file object
        Open file, positioned at (or before) the block to read.

    Returns
    -------
    dict of str to str
        The key/value pairs of the block.  Empty if no pair was found.
    """
    args = {}
    while True:
        pos  = fp.tell()
        raw  = fp.readline()
        if not raw:
            break                # end of file
        line = _decode_if_needed(raw).strip()
        if len(line) == 0:
            continue             # blank line
        if line[0] == '#':
            if len(args) > 0:
                break            # comment after the pairs: end of the block
            continue             # comment before the pairs: block header
        if ':' not in line:
            fp.seek(pos)         # not a pair: end of the block, put it back
            break
        key,val = line.split(':',1)
        args[key.strip()] = val.strip()
    return args

def read(fp, tag):
    pos = fp.tell()
    line = _decode_if_needed(fp.readline().strip())
    while len(line) > 0 and line[0] == "#":
        pos = fp.tell()
        line = _decode_if_needed(fp.readline().strip())
    if not line.startswith(tag+':'):
        fp.seek(pos)
        return None
    return line[(len(tag)+1):].strip()


def write(fp,tag,value,space=False):
    if space:
        fp.write(tag+': '+value+'\n')
    else:
        fp.write(tag+':'+value+'\n')
