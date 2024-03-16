import os, math, json, sys
import numpy as np
import numba, numba.experimental

_Chromosome_spec = [
    ( 'posv', numba.uint32[:] ),
    ( 'datv', numba.uint32[:] ),
    ( 'ncells', numba.uint32 ),
    ( 'chrom_len', numba.uint32 ),
    ( 'dwords_per_site', numba.uint32 ),
    ( 'smoothed', numba.double[:] ) ]

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

    def smooth( self, hw=1000 ):

        self.smoothed = np.zeros( len(self.posv), dtype=np.double )

        counts_u, counts_m = self.count()

        left = 0
        right = 0
        for i in range( len(self.posv) ):

            curpos = float(self.posv[i])

            while left < len(self.posv) and self.posv[left] < curpos - hw:
                left += 1
            right = max( left, right )
            while right < len(self.posv) and self.posv[right] < curpos + hw:
                right += 1

            num = 0
            den = 0
            for j in range( left, right ):            
                dist = ( float(self.posv[j]) - curpos ) / hw
                kernel_weight = ( 1 - abs(dist)**3 )**3
                num += counts_m[j] / ( counts_m[j] + counts_u[j] ) * kernel_weight
                den += kernel_weight
            self.smoothed[i] = ( num + 1 ) / ( den + 1 )





class MetdenseDataset( object ):
    
    def __init__( self, metdense_dir ):

        with open( os.path.join( metdense_dir, "metadata.json" ) ) as f:
            self.info = json.load(f)

        self.ncells = len( self.info["cells"] )

        self.chroms = {}
        for chr in self.info["chromosomes"].keys():
            self.chroms[chr] = Chromosome( metdense_dir, chr, self.ncells, self.info["chromosomes"][chr] )

    def __getitem__( self, idx ):
        chr, site_idx, cell_idx = idx
        return self.chroms[chr].get( site_idx, cell_idx )


md = MetdenseDataset( "scnmt_data__CpG_filtered.metdense" )
print( md[ "Y", 0, 505 ] )

md.chroms["Y"].smooth()
print( md.chroms["Y"].smoothed )

sys.exit(0)

with open( os.path.join( "scnmt_data__CpG_filtered.metdense", "metadata.json" ) ) as f:
    info = json.load(f)

chrY = Chromosome( "scnmt_data__CpG_filtered.metdense", "Y", len( info["cells"] ), -1 )
print( chrY.get( 0, 505 ) )
print( chrY.count() )