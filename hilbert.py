import numpy as np
import numba

@numba.njit
def hilbert( t :int, lv :int ):
    if lv == 0:
        return ( 0, 0 )
    r = 1 << ( 2 * (lv-1) )
    q = t // r
    x, y = hilbert( t % r, lv - 1 )
    if q == 0:
        return y, x
    elif q == 1:
        return x, y + ( 1 << (lv-1) )
    elif q == 2:
        return x + ( 1 << (lv-1) ), y + ( 1 << (lv-1) )
    elif q == 3:
        return (1 << lv) - 1 - y, ( 1 << (lv-1) ) - 1 - x
    else:
        raise SystemError
    
def vec2hilbert( v, lv ):
    img = np.zeros( ( 1<<lv, 1<<lv ) )
    for i in range( v.shape[0] ):
        x, y = hilbert( i, lv )
        img[ x, y ] = v[i]
    return img