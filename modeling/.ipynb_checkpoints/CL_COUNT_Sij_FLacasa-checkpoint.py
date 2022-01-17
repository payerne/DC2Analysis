import numpy as np
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
 
def Sij_FLacasa(Redshift_bin):
    r"""
    Attributes:
    -----------
    Redshift_bin: array
        list of redshift bins
    Returns:
    --------
    Sij: array
        matter fluctuation amplitude in redshift bins
    r"""
    z_arr = np.linspace(0.05,2.5,3000)
    nbins_T   = len(Redshift_bin)
    windows_T = np.zeros((nbins_T,len(z_arr)))
    for i, z_bin in enumerate(Redshift_bin):
        Dz = z_bin[1]-z_bin[0]
        z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
        for k, z in enumerate(z_arr):
            if ((z>z_bin[0]) and (z<=z_bin[1])):
                windows_T[i,k] = 1/Dz  
    Sij = PySSC.Sij(z_arr,windows_T)
    return Sij