import os, math, json
import numpy as np
import numba, numba.experimental

_Chromosome_spec = [
    ( 'posv', numba.uint32[:] ),
    ( 'datv', numba.uint32[:] ),
    ( 'ncells', numba.uint32 ),
    ( 'chrom_len', numba.uint32 ),
    ( 'dwords_per_site', numba.uint32 ) ]

@numba.experimental.jitclass( _Chromosome_spec )
class Chromosome( object ):

    def __init__( self, dir, name, ncells, chrom_len ):
        with numba.objmode():
            self.posv = np.memmap( os.path.join( dir, name + ".pos" ), np.uint32 )
            self.datv = np.memmap( os.path.join( dir, name + ".dat" ), np.uint32 )
            self.ncells = ncells
            self.chrom_len = chrom_len
            self.dwords_per_site = math.ceil( self.ncells / 16 )

    def get( self, site_idx, cell_idx ):
        dword = self.datv[ site_idx * self.dwords_per_site + cell_idx // 16 ]
        dword >>= 2*( cell_idx % 16 )
        call = dword & 3
        return call
    
    def count( self ):
        counts_u = np.zeros( len(self.posv), dtype=np.uint32 )
        counts_m = np.zeros( len(self.posv), dtype=np.uint32 )

        for site_idx in range( len(self.posv) ):
            for cell_idx in range(self.ncells):
                dword = self.datv[ site_idx * self.dwords_per_site + cell_idx // 16 ]
                dword >>= 2*( cell_idx % 16 )
                call = dword & 3
                if call == 3:
                    raise ValueError( "ambiguous call" )
                elif call == 1:
                    counts_u[site_idx] += 1
                elif call == 2:
                    counts_m[site_idx] += 1
                else:
                    assert call == 0

        return counts_u, counts_m



with open( os.path.join( "scnmt_data__CpG_filtered.metdense", "metadata.json" ) ) as f:
    info = json.load(f)

chrY = Chromosome( "scnmt_data__CpG_filtered.metdense", "Y", len( info["cells"] ), -1 )
print( chrY.get( 0, 505 ) )
print( chrY.count() )