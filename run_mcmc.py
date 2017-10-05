from mcmc_lib import *
from noshellavg import *
from run_error_analysis import *
import os
import numpy as np

import argparse
import yaml

def main(params, pool=None):
    run_mcmc(params,pool)

def run_mcmc(params, pool=None):    

    kmin, kmax, kN = params['k']
    rmin, rmax, rN = params['r']
    logscale = params['logscale']
    KMIN, KMAX = 1e-3, 2.
    lmax = params['lmax']
    parameter_ind = params['parameter_ind']  

    (varied_params, varied_params_fid,
     cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma) = parse_priors_and_ranges(params)
       
    b,f,s,nn = cosmo_fid
    print '-----------------------------------'
    print ' Run Error Analaysis'
    print '-----------------------------------'
    print ' parameter setting'
    print ' b={} f={} s={} nn={}'.format(b,f,s,nn)
    print ' k = [{}, {}], kN={}'.format(kmin, kmax, kN)
    print ' r = [{}, {}], rN={}'.format(rmin, rmax, rN)
    print ' lmax={}'.format(lmax)
    print '-----------------------------------'
    
    lik_class = NoShell_covariance_MCMC(KMIN, KMAX, rmin, rmax, 2**10 + 1, rN, kN, b, f, s, nn )


    if 'fisher_filename' not in params : 
        Covariance_matrix(params, lik_class)
        Calculate_Fisher_tot(params, lik_class, kmin = kmin, kmax = kmax, lmax=lmax)
        if 'p' in params['probe']: 
            #print 'calling p fisher'
            params['fisher_filename'] = params['fisher_bandpower_P_filename']
        if 'xi' in params['probe']:
            #print 'calling xi fisher'
            params['fisher_filename'] = params['fisherXi_filename']
        if 'p' in params['probe'] and 'xi' in params['probe']:
            #print 'calling p+xi fisher'
            params['fisher_filename'] = params['fishertot_filename']     
    fisher_filename = params['fisher_filename']
    
    
    if 'mask_filename' not in params :   
        maskP = np.ones(lik_class.kcenter_y.size * 3, dtype=bool)
        maskX = np.ones(lik_class.rcenter.size * 3, dtype=bool)
        maskP = generate_mask_datav(lik_class, maskP, kmin = kmin, kmax=kmax, lmax=lmax, xi=False).ravel() 
        maskX = generate_mask_datav(lik_class, maskX, kmin = kmin, kmax=kmax, lmax=lmax, xi=True).ravel() 
        
        if 'p' not in params['probe']: 
            maskP = np.zeros(lik_class.kcenter_y.size * 3, dtype=bool)
        if 'xi' not in params['probe']: 
            maskX = np.zeros(lik_class.rcenter.size * 3, dtype=bool)

        mask = np.hstack([maskP, maskX])

        f = 'data_txt/datav/'+params['name']+'.mask'
        np.savetxt(f,mask)
        params['mask_filename'] = f
    mask_filename = params['mask_filename']
        
        
    if 'datav_filename' not in params :
        datavP = P_multipole(lik_class)
        datavXi = Xi_multipole(lik_class)
        #datavP = np.genfromtxt(params['multipole_p_filename'])
        #datavXi = np.genfromtxt(params['multipole_xi_filename'])
        #datavP = masking_paramsdatav(lik_class, datavP, kmin = kmin, kmax=kmax, lmax=lmax)
        #datavXi = masking_paramsdatav(lik_class, datavXi, kmin = kmin, kmax=kmax, lmax=lmax, xi=True) 
        datav = np.hstack([datavP, datavXi])
        f = 'data_txt/datav/'+params['name']+'.datavector'
        np.savetxt(f,datav)
        
        params['datav_filename'] = f
        
    datav_filename = params['datav_filename']
      
        
    print '-----------------------------------'
    print ' MCMC Settings'
    print '-----------------------------------'
    
    lik_class.init_like_class(datav_filename, fisher_filename, mask_filename)
    
    print " will sample over ", varied_params
    print " fiducial values =", varied_params_fid
    
    nthreads = 32
    if 'n_threads' in params:
        nthreads = params['n_threads']
    iterations = 10000
    if 'iterations' in params:
        iterations = params['iterations']
    nwalker = 32
    if 'nwalker' in params:
        nwalker = params['nwalker']

    chain_file = 'like/like_'+params['name'] + '_sam{}'.format(iterations * nwalker)

    print ' estimator :', params['probe'] 
    print ' iterations {}'.format(iterations)
    print ' nthreads   {}'.format(nthreads)
    print ' nwalker    {}'.format(nwalker)
    print ' chains will be stored in :', chain_file
    print '-----------------------------------'
    
    
    sample_main(lik_class, varied_params, iterations, nwalker, nthreads, chain_file,
    cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, pool=pool)
  

    
def parse_priors_and_ranges(params):


    cosmo_names = ['b', 'f', 's', 'nn']
    varied_params = []
    varied_params_fid = []
    cosmo_min = []
    cosmo_fid = []
    cosmo_max = []
    cosmo_fid_sigma=[]
    for p in cosmo_names:
        p_range = params[p+"_range"]
        min_val, fid_val, max_val, sigma, is_var = parse_range(p_range)
        if is_var:
            varied_params.append(p)
            varied_params_fid.append(fid_val)
    # always set min, fid, max values. 
    # the min/max will only actually be referenced if the 
    # parameter is varied.
    # If the parameter is varied then fid acts as a starting
    # point for the sampler, if not then the value is fixed there.
        cosmo_min.append(min_val)
        cosmo_fid.append(fid_val)
        cosmo_max.append(max_val)
        cosmo_fid_sigma.append(sigma)

    varied_params_fid = np.array(varied_params_fid)
    cosmo_min = np.array(cosmo_min)
    cosmo_fid = np.array(cosmo_fid)
    cosmo_max = np.array(cosmo_max)
    cosmo_fid_sigma = np.array(cosmo_fid_sigma)
    
    return (varied_params, varied_params_fid,
            cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma)


def parse_range(p_range):
    "return min, fid, max, is_varied"
    if np.isscalar(p_range):
        min_val = p_range
        fid_val = p_range
        max_val = p_range
        sig_val = p_range
        is_var  = False
    elif len(p_range)==1:
        min_val = p_range[0]
        fid_val = p_range[0]
        max_val = p_range[0]
        sig_val = p_range[0]
        is_var  = False
    else:
        if len(p_range)!=4:
            raise ValueError("Must specify 1 or 3 elements in param ranges")
        min_val = p_range[0]
        fid_val = p_range[1]
        max_val = p_range[2]
        sig_val = p_range[3]
        is_var  = True

    return min_val, fid_val, max_val, sig_val, is_var


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='call run_error_analysis outside the pipeline')
    parser.add_argument("parameter_file", help="YAML configuration file")
    args = parser.parse_args()
    try:
        param_file = args.parameter_file   
    except SystemExit:
        sys.exit(1)

    params = yaml.load(open(param_file))
    run_mcmc(params)

