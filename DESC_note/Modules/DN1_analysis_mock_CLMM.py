import numpy as np
import pickle, sys
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock

def load(filename, **kwargs):
    
    """Loads GalaxyCluster object to filename using Pickle"""
    
    with open(filename, 'rb') as fin:
        
        return pickle.load(fin, **kwargs)

cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

where_catalogs = '/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/mock_CLMM/'

cluster_infos = load('/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/mock_CLMM/mock_cluster.pkl')

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

Mass_bin = [[10**12, 10**18]]
Z_bin = [[0.2, 1]]

r"""
check catalogs
"""

redshift_def = 'redshift'
Obs_bin = Mass_bin
obs_def = 'M200c'
cluster_name = 'id'
cluster_key = 'cluster_id_mock_'
column_to_extract = ['halo_id', 'sigma_c_1_TRUE', 'et', 'ex']

r"""
make_Galaxy_Cluster_Catalog
"""

z_gal_name = 'z'
cluster_ra = 'ra'
cluster_dec = 'dec'
shape_component1_in = 'e1'
shape_component2_in = 'e2'                                           
tan_component_out = 'et'
cross_component_out = 'ex'
rmax = 6
def quantity_modifier(cl): 
    
    print('1')
    
    cosmo.eval_sigma_crit(cl.z, cl.galcat[z_gal_name])
    
    cl.galcat['sigma_crit^2'] = cosmo.eval_sigma_crit(cl.z, cl.galcat[z_gal_name])**2
    
    return cl

modify_quantity = True
compute_selection_response = False
quantity_modifier = quantity_modifier
sigma_c_1_in =  'sigma_c_1_TRUE'
err_shape_in = None

r"""
Estimating shear profile
"""

down, up, n_bins = 0.1 , rmax, 20
bin_edges = make_bins(down, up, nbins=n_bins, method='evenlog10width')
radial_bin = binning(bin_edges)
col_to_bin = ['w_ls','halo_id','st', 'sx', 'r',]

col_to_average = ['st', 'sx', 'r']
name_averaged_cl = ['gt_av', 'gx_av', 'radius_av',]

col_to_sum = ['w_ls']
name_sum_cl = ['norm_sum']

extract_extra = ['n_gal']

r"""
Saving file
"""
save= True

where_to_save = '/pbs/throng/lsst/users/cpayerne/ThesisAtCCin2p3/Galaxy_Cluster_Catalogs_details/analysis_mock_CLMM/'

filename = 'mock_CLMM.pkl'

name = where_to_save + filename



