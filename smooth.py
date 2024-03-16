import math, json
import numpy as np
import numba

@numba.njit
def count( pos, dat ):
    counts_u = np.zeros( len(pos), dtype=np.uint32 )
    counts_m = np.zeros( len(pos), dtype=np.uint32 )

    for site_idx in range( len(pos) ):
        for cell_idx in range(ncells):
            dword = dat[ site_idx * dwords_per_site + cell_idx // 16 ]
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


with open( "testdata/metadata.json" ) as f:
    info = json.load(f)

ncells = len(info["cells"] )    
dwords_per_site = math.ceil( ncells / 16 )

pos = np.memmap( "scnmt_data__CpG_filtered.metdense/1.pos", np.uint32 )
dat = np.memmap( "scnmt_data__CpG_filtered.metdense/1.dat", np.uint32 )

counts_u, counts_m = count( pos, dat )

print( counts_u, counts_m )    