import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/lenspack/')
from lenspack.utils import bin2d
from lenspack.image.inversion import ks93
from lenspack.geometry.projections.gnom import radec2xy, xy2radec
import numpy as np
from clmm.utils import compute_lensed_ellipticity
from scipy.interpolate import RectBivariateSpline

def compute_ellipticity_from_lensing_map(z_cl, z_gal_0, 
                        ra_gal, dec_gal, z_gal, 
                        shear1_map, shear2_map, kappa_map, 
                        shapenoise = None, cosmo = None):
    r"""
    Attributes:
    -----------
    z_cl: float
        cluster redshift
    z_gal_0: float
        default p-300 galaxy redshift
    ra_gal: array
        galaxy right ascensions
    dec_gal: array
        galaxy declinaisons
    z_gal: array
        galaxy redshifts
    shear1_map: fct
        map of shear1
    shear2_map: fct
        map of shear2
    kappa_map: fct
        map of kappa
    shapenoise: float
        intrinsic ellipticity shapenoise
    cosmo: Cosmology object (CLMM)
        cosmology object
    Returns:
    --------
    e1_lensed, e2_lensed: array, array
        lensed ellipticity components 1 & 2
    """
    sigma_crit_z_cl_z_gal_0 = cosmo.eval_sigma_crit(z_cl, z_gal_0)
    sigma_crit_z_cl_z_gal = cosmo.eval_sigma_crit(z_cl, z_gal)
    rescale = sigma_crit_z_cl_z_gal_0/sigma_crit_z_cl_z_gal 
    #rescaling kappa, shear1, shear2
    kappa_gal = kappa_map(ra_gal, dec_gal, grid = False) * rescale
    shear1_gal = shear1_map(ra_gal, dec_gal, grid = False) * rescale
    shear2_gal = shear2_map(ra_gal, dec_gal, grid = False) * rescale

    random_e1, random_e2 = np.random.randn(2)
    e1_gal_true = random_e1 * shapenoise
    e2_gal_true = random_e2 * shapenoise
    e1_lensed, e2_lensed = compute_lensed_ellipticity(e1_gal_true, e2_gal_true, 
                                                      shear1_gal, shear2_gal, kappa_gal)
    return e1_lensed, e2_lensed

def compute_lensing_map_from_ellipticity(ra_gal, dec_gal, 
                                           e1_gal, e2_gal, 
                                           resolution = .3, 
                                           filter_resolution = None):
    r"""
    Attributes:
    -----------
    ra_gal: array
        galaxy right ascensions
    dec_gal: array
        galaxy declinaisons
    e1_gal: array
        galaxy ellipticity (1)
    e2_gal: array
        galaxy ellipticity (2)
    resolution: float
        resolution for ks93
    Returns:
    --------
    X, Y: array, array
        right ascension and declinaison of kappas map
    kappaE, kappaB: array, array
        kappa maps from ellipticities
    """
    ra_mean = np.mean(ra_gal)
    dec_mean = np.mean(dec_gal)
    # Projection all objects from spherical to Cartesian coordinates
    x, y =  radec2xy(ra_mean, dec_mean, ra_gal, dec_gal)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_x_deg = np.rad2deg(size_x)
    size_y_deg = np.rad2deg(size_y)
    Nx = int(size_x_deg / resolution * 60)
    Ny = int(size_y_deg / resolution * 60)
    x_bin = np.linspace(min_x, max_x, Nx)
    y_bin = np.linspace(min_y, max_y, Ny)
    ra_bin, dec_bin = xy2radec(0, 0, x_bin, y_bin)
    RA_bin = ra_bin
    DEC_bin = dec_bin
    X, Y = np.meshgrid(RA_bin, DEC_bin)
    g1_tmp, g2_tmp = bin2d(x,y, npix=(Nx, Ny), v=(e1_gal, e2_gal), 
                           extent=(min_x, max_x, min_y, max_y))
    g_corr_mc_ngmix_map = np.array([g1_tmp, g2_tmp])
    kappaE, kappaB = ks93(g_corr_mc_ngmix_map[0], -g_corr_mc_ngmix_map[1])
    if filter_resolution != None:
        kappaE = gaussian_filter(kappaE, filter_resolution)
        kappaB = gaussian_filter(kappaB, filter_resolution)
    return X, Y, kappaE, kappaB

def interp_shear_kappa_map(shear1, shear2, kappa, ra, dec):
    r"""
    Attributes:
    -----------
    kappa: array
        tabulated kappa map
    shear1: array
        tabulated shear1 map
    shear2: array
        tabulated shear2 map
    ra: array
        ra axis used for tabulation
    dec: array
        dec axis used for tabulation
    Returns:
    --------
    kappa_map: fct
        interpolated kappa map
    shear1_map: fct
        interpolated shear1 map
    shear2_map: fct
        interpolated shear2 map
    """
    shear1_map = interp(np.sort(ra), np.sort(dec), shear1)
    shear2_map = interp(np.sort(ra), np.sort(dec), shear2)
    kappa_map  = interp(np.sort(ra), np.sort(dec), kappa)
    return shear1_map, shear2_map, kappa_map

def interp(ra_array, dec_array, map):
    r"""
    Attributes:
    -----------
    ra_array, dec_array : array, array
        ra, dec axis of the map
    map: array
        2D map
    Returns:
    --------
    fct : function
        interpolated 2D map
    """
    return RectBivariateSpline(ra_array, dec_array, map)