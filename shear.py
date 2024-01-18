import numpy as np
import math
from numba import jit
@jit(nopython=True)
def usts(u,usta,tausta,fr,ie,hs_up,g,sd,nx):
    for i in np.arange(0,nx+1):
        ri=hs_up[i]*ie[i]
        usta[i]=math.sqrt(g*ri)
        tausta[i]=ri/sd
        fr[i]=u[i]/math.sqrt(hs_up[i]*g)
    return usta,tausta,fr