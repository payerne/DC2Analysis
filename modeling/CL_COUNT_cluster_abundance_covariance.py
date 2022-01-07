def covariance_SSC(self, Z_bin, logMass_bin, cosmo):
        
        N_th_true_cosmo = self.model_Binned_Nth(Z_bin, logMass_bin, cosmo)
        halo_biais_true_cosmo = self.model_Binned_halo_biais(Z_bin, logMass_bin, cosmo, N_th_true_cosmo).flatten()
        massdef_new = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
        hmd_new = ccl.halos.MassFuncDespali16(cosmo, mass_def=massdef_new)
        halobiais = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def=massdef_new, mass_def_strict=True)
        z_arr = np.linspace(0.05,2.5,3000)
        nbins_T   = len(Z_bin)
        windows_T = np.zeros((nbins_T,len(z_arr)))
        for i, z_bin in enumerate(Z_bin):
            Dz = z_bin[1]-z_bin[0]
            z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
            for k, z in enumerate(z_arr):
                if ((z>z_bin[0]) and (z<=z_bin[1])):
                    windows_T[i,k] = 1/Dz  
        Sij = PySSC.Sij(z_arr,windows_T)
        LogM, Z = np.meshgrid(np.mean(logMass_bin, axis = 1), np.mean(Z_bin, axis = 1))
        index_LogM, index_Z =  np.meshgrid(np.arange(len(logMass_bin)), np.arange(len(Z_bin)))
        len_mat = len(Z_bin) * len(logMass_bin)
        cov_SSC = np.zeros([len_mat, len_mat])
        for i, Ni in enumerate(N_th_true_cosmo.flatten()):
            index_z_i = index_Z.flatten()[i]
            index_logm_i = index_LogM.flatten()[i]
            logm_mean_i = np.mean([LogM.flatten()[index_logm_i], LogM.flatten()[index_logm_i + 1]])
            z_mean_i = np.mean([Z.flatten()[index_z_i], Z.flatten()[index_z_i + 1]])
            for j, Nj in enumerate(N_th_true_cosmo.flatten()):
                index_z_j = index_Z.flatten()[j]
                index_logm_j = index_LogM.flatten()[j]
                logm_mean_j = np.mean([LogM.flatten()[index_logm_j], LogM.flatten()[index_logm_j + 1]])
                z_mean_j = np.mean([Z.flatten()[index_z_j], Z.flatten()[index_z_j + 1]])
                hbi = halobiais.get_halo_bias(cosmo, 10**logm_mean_i, 1./(1. + z_mean_i), mdef_other = massdef_new)
                hbj = halobiais.get_halo_bias(cosmo, 10**logm_mean_j, 1./(1. + z_mean_j), mdef_other = massdef_new)
                hbi = halo_biais_true_cosmo[i]
                hbj = halo_biais_true_cosmo[j]
               # print(hbj_)
                #print(' ')
                cov_SSC[i,j] = hbi * hbj * Ni * Nj * Sij[index_z_i,index_z_j]
        return Sij, cov_SSC/self.CA.f_sky