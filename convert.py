# This script converts the output of scbs into the 'metdense' directory format

import sys
import os
import math
import struct
import yaml, json
import numpy as np
import numba

@numba.njit
def convert_npz_inner( indptr, indices, data, i, buf ):
    buf[:] = 0
    cell_idx = 0
    for j in range( indptr[i], indptr[i+1] ):
        if data[j] == -1:
            w = 1
        elif data[j] == 1:
            w = 2
        else:
            raise RuntimeError( "Invalid value in npz file." )
        w <<= ( indices[j] % 16 ) * 2
        buf[ indices[j] // 16 ] |= w
        cell_idx += 1


def convert_npz( npz_filename, outdir, out_filename_prefix, ncells ):

    npz = np.load( npz_filename )

    if "format" not in npz or npz["format"].item() != b'csr':
        raise RuntimeError( npz_filename + ": npz file is not a scipy sparse matrix in CSR format." )
    
    chromlen, ncells_ = npz["shape"]
    if ncells_ != ncells:
        raise RuntimeError( npz_filename + ": Inconsistent number of cells." )

    indices = npz["indices"] 
    indptr = npz["indptr"]
    data = npz["data"]

    fpos = open( os.path.join( outdir, out_filename_prefix + ".pos" ), "wb" )
    fdat = open( os.path.join( outdir, out_filename_prefix + ".dat" ), "wb" )

    buf = np.zeros( math.ceil( ncells/16 ), np.uint32 )

    for i in range( len(indptr) - 1 ): 
        if (i+1) % 10000000 == 0:
            print( out_filename_prefix + ":", format( i+1, "," ) )
        if indptr[i] == indptr[i+1]:
            continue

        fpos.write( struct.pack( 'I', i-1 ) )    # here, we fix that the old format is one-based
        convert_npz_inner( indptr, indices, data, i, buf )
        buf.tofile( fdat )

    fdat.close()
    fpos.close()

    print( out_filename_prefix + ":", format( chromlen, "," ) )
    return int(chromlen)


indir = sys.argv[1] #"/home/anders/tmp/scbs_npz"
outdir = sys.argv[2] #"testdata"
genome_id = sys.argv[3]

with open( os.path.join( indir, "column_header.txt" ) ) as f:
    cellnames = f.readlines()
cellnames = [ cellname.rstrip() for cellname in cellnames ]

chromlens = dict()
for fn in os.listdir( indir ):
    if fn.endswith( ".npz" ):
        chromname = fn[:-4]
        if chromname not in ( "Y", "MT" ):
            continue
        fullpath = os.path.join( indir, fn )
        chromlen = convert_npz( fullpath, outdir, chromname, len(cellnames) )
        chromlens[ chromname ] = chromlen

with open( os.path.join( outdir, "metadata.json" ), "w" ) as f:
    json.dump( 
        { "format": "metdense", 
          "format-version": "0.0.-1", 
          "genome": genome_id, 
          "cells": cellnames, 
          "chromosomes": chromlens },
        f, indent=3 )

print( "finished" )