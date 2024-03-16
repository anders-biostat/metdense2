import math
import numpy as np

pos = np.memmap( "testdata/Y.pos", np.uint32 )
dat = np.memmap( "testdata/Y.dat", np.uint32 )
ncells = 833
dwpp = math.ceil( ncells / 16 )

for i in range(10000):
    print( pos[i], ": ", end="" )
    for j in range(ncells):
        w = ( dat[ i*dwpp + j//16 ] >> ( 2 * (j%16) ) ) & 3
        assert w in ( 0, 1, 2 )
        print( "_UM"[w], end="" )
    print()

