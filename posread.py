from calendar import WEDNESDAY
import numpy as np
import math,csv
from numba import jit

def pos(bed_file):
    fopen=open(bed_file,'r')
    dataReader=csv.reader(fopen)
    d1=next(dataReader); npos=int(d1[0])
    xpos=np.zeros(npos,dtype=float)
    zpos=np.zeros_like(xpos); bpos=np.zeros_like(xpos)
    for n in np.arange(0,npos):
        lp=next(dataReader)
        xpos[n]=float(lp[0]);zpos[n]=float(lp[1]);bpos[n]=float(lp[2])
    slope1=(zpos[0]-zpos[1])/(xpos[1]-xpos[0])
    slope2=(zpos[npos-2]-zpos[npos-1])/(xpos[npos-1]-xpos[npos-2])
#   print(1./slope1,1./slope2);exit()
    return npos,xpos,zpos,bpos,slope1,slope2
