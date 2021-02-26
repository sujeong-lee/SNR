import os, sys
sys.path.append('src/')
from mcmc_lib import *
#from noshellavg import *
from discrete import *
from run_error_analysis import *
import numpy as np

import argparse
import yaml

def main(params, pool=None):
    run_mcmc(params,pool)

def run_mcmc(params, pool=None):    

    #save_dir = 'output/'+params['name']+'/'
    #params['savedir'] = params['savedir']+'/chain/'
    save_dir = params['savedir']+'/chain/'
    if os.path.exists(save_dir): 
        print 'savedir exists...', save_dir
    else : 
        os.makedirs(save_dir)
        print 'create save directory..', save_dir

    KMIN, KMAX = 1e-4, 10
    if 'KRANGE_Fourier' in params : 
        KMIN, KMAX = params['KRANGE_Fourier']

    kmin, kmax, kN = params['k']
    rmin, rmax, rN = params['r']

    kscale = 'log'
    if 'kscale' in params : kscale = params['kscale']
    rscale = 'lin'
    if 'rscale' in params : rscale = params['rscale']
    
    lmax = params['lmax']
    parameter_ind = params['parameter_ind']  
    probe = params['probe']

    (varied_params, varied_params_fid,
     cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma) = parse_priors_and_ranges(params)
     
    b,f,s,nn = cosmo_fid

    print '-----------------------------------'
    print ' Run Error Analaysis'
    print '-----------------------------------'
    print ' jobname :', params['name']
    print ' parameter setting'
    print ' probe=', probe
    print ' b={} f={} s={} nn={}'.format(b,f,s,nn)
    print ' k = [{}, {}], kN={}'.format(kmin, kmax, kN)
    print ' r = [{}, {}], rN={}'.format(rmin, rmax, rN)
    print ' lmax={}'.format(lmax)
    print '-----------------------------------'

    lik_class = covariance_MCMC(KMIN, KMAX, rmin, rmax, 20000, kN, rN, b, f, s, nn, kscale = kscale, rscale=rscale )

    if 'matterpower' in params :
        file = params['matterpower']
    else : 
        file = 'src/matterpower_z_0.55.dat'  # from camb (z=0.55)

    lik_class.mPk_file = file
    print 'calling stored matter power spectrum.. ', file


    if 'generate_datav_only' not in params : 
        if 'fisher_filename' not in params : 
            Covariance_matrix(params, lik_class)
            #P_multipole(lik_class)
            Calculate_Fisher_tot(params, lik_class, kmin = kmin, kmax = kmax, lmax=lmax)
            #params_datavector(params, lik_class)
            #params_xi_datavector(params, lik_class)
            
            if 'p' in params['probe']: 
                #print 'calling p fisher'
                params['fisher_filename'] = params['fisher_bandpower_P_filename']
            if 'xi' in params['probe']:
                #print 'calling xi fisher'
                params['fisher_filename'] = params['fisherXi_filename']
            if 'p' in params['probe'] and 'xi' in params['probe']:
                print 'calling p+xi fisher'
                params['fisher_filename'] = params['fishertot_filename']  
        else : 
            P_multipole(params, lik_class)
            #lik_class.multipole_P_band_all()
            
        fisher_filename = params['fisher_filename']
    
    elif 'generate_datav_only' in params : 
        if params['generate_datav_only'] : P_multipole(params, lik_class)
        else : pass

    
    if 'mask_filename' not in params :   
        maskP = np.ones(lik_class.kcenter_y.size * 3, dtype=bool)
        maskX = np.ones(lik_class.rcenter.size * 3, dtype=bool)
        maskP = generate_mask_datav(lik_class, maskP, kmin = kmin, kmax=kmax, lmax=lmax, xi=False).ravel() 
        maskX = generate_mask_datav(lik_class, maskX, kmin = kmin, kmax=kmax, lmax=lmax, xi=True).ravel() 
        maskPzero = np.zeros(lik_class.kcenter_y.size * 3, dtype=bool)
 	maskXzero = np.zeros(lik_class.rcenter.size * 3, dtype=bool)

	mask_com = np.hstack([maskP, maskX])
	np.savetxt(save_dir + 'mask_com.txt',mask_com)	

        if 'p' not in params['probe']: 
            maskP = maskPzero #np.zeros(lik_class.kcenter_y.size * 3, dtype=bool)
        if 'xi' not in params['probe']: 
            maskX = maskXzero #np.zeros(lik_class.rcenter.size * 3, dtype=bool)

        mask = np.hstack([maskP, maskX])
	mask_onlyP = np.hstack([maskP, maskXzero])
	mask_onlyX = np.hstack([maskPzero, maskX])
        print 'masked points :', np.sum(~mask)
        #f = 'data_txt/datav/'+params['name']+'.mask'
        f = save_dir + 'mask.txt'
        np.savetxt(f,mask)
	np.savetxt(save_dir + 'mask_p.txt',mask_onlyP)
	np.savetxt(save_dir + 'mask_xi.txt',mask_onlyX)
	np.savetxt(f,mask)
        params['mask_filename'] = f
    mask_filename = params['mask_filename']
        
        
    

    if 'datav_filename' not in params :
        datavP = P_multipole(params, lik_class)
        datavXi = Xi_multipole(params, lik_class)
        #datavP = np.genfromtxt(params['multipole_p_filename'])
        #datavXi = np.genfromtxt(params['multipole_xi_filename'])
        #datavP = masking_paramsdatav(lik_class, datavP, kmin = kmin, kmax=kmax, lmax=lmax)
        #datavXi = masking_paramsdatav(lik_class, datavXi, kmin = kmin, kmax=kmax, lmax=lmax, xi=True) 
        datav = np.hstack([datavP, datavXi])

        #f = 'data_txt/datav/'+params['name']+'.datavector'
        f = save_dir+'datavector.txt'
        #f = savedir + 'PXi.datavector'

        np.savetxt(f,datav)
        
        params['datav_filename'] = f
        
    datav_filename = params['datav_filename']
      
       
    if 'generate_datav_only' in params : 
        if params['generate_datav_only'] : 
            Nr = lik_class.rcenter.size
            Nk = lik_class.kcenter_y.size
            datavXi = np.column_stack(( lik_class.rcenter, datavXi[:Nr], datavXi[Nr:Nr*2],datavXi[Nr*2:Nr*3] ))
            datavP = np.column_stack(( lik_class.kcenter_y, datavP[:Nk], datavP[Nk:Nk*2], datavP[Nk*2:Nk*3] ))
            np.savetxt(save_dir + 'Xi.txt' ,datavXi, header= 'r, xi0, xi2, xi4')
            np.savetxt(save_dir + 'P.txt' ,datavP, header = 'k, p0, p2, p4')
            print 'datav saved to ', datav_filename
            print 'datav saved to ', save_dir + 'P.txt' 
            print 'datav saved to ', save_dir + 'Xi.txt'
            print 'datav saved to ', save_dir + 'mask.txt'
	    
            return 0
        else : pass
    else : pass

    lik_class.init_like_class(datav_filename, fisher_filename, mask_filename, probe=probe)
    

    if params['sampler'] == 'emcee':

        print '-----------------------------------'
        print ' MCMC Settings'
        print '-----------------------------------'

        print " sampler:", params['sampler']
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

        chain_file = save_dir + 'chain_'+params['name']+'_sam{}'.format(iterations * nwalker)

        print ' estimator :', params['probe'] 
        print ' iterations {}'.format(iterations)
        print ' nthreads   {}'.format(nthreads)
        print ' nwalker    {}'.format(nwalker)
        print ' chains will be stored in :', chain_file
        print '-----------------------------------'
        
        
        sample_main(lik_class, varied_params, iterations, nwalker, nthreads, chain_file,
        cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, pool=pool)
  

    if params['sampler'] == 'snake':

        print '-----------------------------------'
        print ' MCMC Settings'
        print '-----------------------------------'

        print " sampler:", params['sampler']

        nsample_dimension = 1000
        if 'nsample_dimension' in params:
            nsample_dimension = params['nsample_dimension']
        threshold = 4
        if 'threshold' in params:
            threshold = params['threshold']
        maxiter = 100000
        if 'maxiter' in params:
            maxiter = params['maxiter']
        chain_file = save_dir + 'snake_'+params['name'] #+'_sam{}'.format(iterations * nwalker)

        print ' estimator :', params['probe'] 
        print ' nsample_dimension {}'.format(nsample_dimension)
        print ' threshold   {}'.format(threshold)
        print ' maxiter    {}'.format(maxiter)
        print ' chains will be stored in :', chain_file
        print '-----------------------------------'

        snake_sampler_main(lik_class, varied_params, None, None, None, 
                chain_file, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
                nsample_dimension, threshold, maxiter, pool=pool)



    if params['sampler'] == 'grid':

        print '-----------------------------------'
        print ' MCMC Settings'
        print '-----------------------------------'

        print " sampler:", params['sampler']

        nsample_dimension = 10
        if 'nsample_dimension' in params:
            nsample_dimension = params['nsample_dimension']
        #nstep = -1
        #if 'nstep' in params:
        #    nstep = params['nstep']
        #maxiter = 100000
        #if 'maxiter' in params:
        #    maxiter = params['maxiter']
        chain_file = save_dir + 'grid_'+params['name']+'.txt' #+'_sam{}'.format(iterations * nwalker)

        print ' estimator :', params['probe'] 
        print ' nsample_dimension {}'.format(nsample_dimension)
        #print ' nstep   {}'.format(nstep)
        #print ' maxiter    {}'.format(maxiter)
        print ' chains will be stored in :', chain_file
        print '-----------------------------------'

        grid_sampler_main(lik_class, varied_params, None, None, None, 
                chain_file, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
                nsample_dimension, None, None, pool=pool)


    if params['sampler'] == 'fmin':

        print '-----------------------------------'
        print ' MCMC Settings'
        print '-----------------------------------'

        print " sampler:", params['sampler']

        #nsample_dimension = 10
        #if 'nsample_dimension' in params:
        #    nsample_dimension = params['nsample_dimension']
        #nstep = -1
        #if 'nstep' in params:
        #    nstep = params['nstep']
        #maxiter = 100000
        #if 'maxiter' in params:
        #    maxiter = params['maxiter']
        save_dir = params['savedir'] + '/fmin/'
        os.system('mkdir '+save_dir)
        chain_file = save_dir + 'fmin_'+params['name']+'.txt' #+'_sam{}'.format(iterations * nwalker)

        print ' estimator :', params['probe'] 
        #print ' nsample_dimension {}'.format(nsample_dimension)
        #print ' nstep   {}'.format(nstep)
        #print ' maxiter    {}'.format(maxiter)
        print ' chains will be stored in :', chain_file
        print '-----------------------------------'

        fmin_sampler_main(lik_class, varied_params, None, None, None, 
                chain_file, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
                None, None, None, pool=pool)


    
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


def mock_pararrel_processing( params, pool=None ):

    n_mocks = params['n_mocks']
    mock_dir = params['mock_dir']
    #mock_probe = params['mock_probe']
    if 'p' in params['probe'] : probe = 'p'
    if 'xi' in params['probe'] : probe = 'xi'


    paramslist = [ params.copy() for i in range( n_mocks )]
    for i in range( 1, n_mocks+1 ):
        
        filename = mock_dir + 'mock_pxi_no{:04d}.txt'.format(i)
        jobname = 'mock_'+probe+'_no{:04d}'.format(i)
        #print 'calling mock datav :', filename
        paramslist[i-1]['name'] = jobname
        paramslist[i-1]['datav_filename'] = filename

    jobs = paramslist

    #Actually compute the likelihood results
    if pool:
        results = pool.map(run_mcmc, jobs )
    else:
        results = list(map(run_mcmc, jobs))

    #if results == None: return 0
    #else : pass


def generate_mock_covariance( params ):

    # call He-eJong's Mock
    savedir = params['savedir']
    n_mocks = params['n_mocks']
    mockdir = params['mock_dir']
    #mock_probe = params['mock_probe']
    #if 'p' in params['probe'] : probe = 'p'
    #if 'xi' in params['probe'] : probe = 'xi'
    if not os.path.exists(savedir+'/chain/mask.txt'): 
	print 'No mask file exists.'
	print 'Generate data vectors...'
	run_mcmc(params)
	return 0

    if 'mock_covtot' in params:
	print 'load mock_covtot...', params['mock_covtot']
        mock_covtot = np.genfromtxt(params['mock_covtot'])


    else : 
	print 'generate covariance matrix from mocks'
        print 'mock dir=', mockdir

        #mockdir = '../data_txt/Heejong/mockdatavectors/reformat_dv_r50_200/'
        mocks_filename = os.listdir(mockdir)
        mocks_filename.sort()
        print 'number of mocks: ', len(mocks_filename)-2     

        mock_datav = []
        i=-1
        for fd in mocks_filename:
            if fd in ['kbin.txt', 'rbin.txt']: pass
            else : mock_datav.append( np.genfromtxt(mockdir + fd) )
            print 'calling mocks {}/{}      \r'.format(i,len(mocks_filename)-2),
            i+=1
        mock_datav = np.array(mock_datav)
        print ''


        # make covariance
        from mock_test import mock_covariance
        mock_covtot = mock_covariance(mock_datav, mock_datav)
        np.savetxt(savedir+'mock_covtot.txt', mock_covtot)
        print 'mock_covtot saved to ', savedir + 'mock_covtot.txt'

    print 'mock_covtot =', mock_covtot.shape


    # cov matrix mask

    maskp = np.genfromtxt(savedir + '/chain/mask_p.txt')
    maskx = np.genfromtxt(savedir +'/chain/mask_xi.txt')
    maskcom = np.genfromtxt(savedir +'/chain/mask_com.txt')
    maskp = np.array(maskp, dtype=bool)
    maskx = np.array(maskx, dtype=bool)
    maskcom = np.array(maskcom, dtype=bool)

    Np = np.sum(maskp)
    Nx = np.sum(maskx)
    Nc = np.sum(maskcom)

    mx, my = np.mgrid[0:maskp.size, 0:maskp.size]

    maskp_2d = maskp[mx] * maskp[my]
    maskx_2d = maskx[mx] * maskx[my]
    maskcom_2d = maskcom[mx] * maskcom[my]
    print 'masking...'
    print 'Np, Nx, Nc = ', Np, Nx, Nc

    mock_covp = mock_covtot[maskp_2d].reshape(Np, Np)
    mock_covxi = mock_covtot[maskx_2d].reshape(Nx, Nx)
    mock_covcom = mock_covtot[maskcom_2d].reshape(Nc, Nc)
    mock_covp = 0.5 * (mock_covp + mock_covp.T)
    mock_covxi = 0.5 * (mock_covxi + mock_covxi.T)
    mock_covtot = 0.5 * (mock_covtot + mock_covtot.T)

    from numpy.linalg import inv
    print 'iverting cov...'
    fisherp = inv(mock_covp)
    fisherxi = inv(mock_covxi)
    fishercom = inv(mock_covcom)

    np.savetxt(savedir+'mockP.fisher', fisherp)
    np.savetxt(savedir+'mockXi.fisher', fisherxi)
    np.savetxt(savedir+'mockcom.fisher', fishercom)

    print 'mock fisher saved to ', basedir+'mockP.fisher'
    print 'mock fisher saved to ', basedir+'mockXi.fisher'
    print 'mock fisher saved to ', basedir+'mockcom.fisher'



def mock_pararrel_processing2( params, pool=None ):

    #n_mocks = params['n_mocks']
    mock_dir = params['mock_dir']
    #mock_probe = params['mock_probe']
    if 'p' in params['probe'] : probe = 'p'
    if 'xi' in params['probe'] : probe = 'xi'


    _filenamelist = os.listdir(mock_dir)
    _filenamelist.sort()
    
    filenamelist = []
    for fn in _filenamelist:
	if fn not in [ 'rbin.txt', 'kbin.txt'] : filenamelist.append(fn)  

    n_mocks = len(filenamelist)
    paramslist = [ params.copy() for i in range( n_mocks )]


    for i in range(1, n_mocks+1 ):
	
	filename = mock_dir + filenamelist[i-1]
	jobname = filenamelist[i-1]
        #filename = mock_dir + 'mock_pxi_no{:04d}.txt'.format(i)
        #jobname = 'mock_'+probe+'_no{:04d}'.format(i)
        
	#print 'calling mock datav :', filename
        paramslist[i-1]['name'] = jobname
        paramslist[i-1]['datav_filename'] = filename

    jobs = paramslist

    #Actually compute the likelihood results
    if pool:
        results = pool.map(run_mcmc, jobs )
    else:
        results = list(map(run_mcmc, jobs))

    #if results == None: return 0
    #else : pass



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='call run_error_analysis outside the pipeline')
    parser.add_argument("parameter_file", help="YAML configuration file")
    args = parser.parse_args()
    try:
        param_file = args.parameter_file   
    except SystemExit:
        sys.exit(1)

    params = yaml.load(open(param_file))

    # Initialize the MPI pool
    from schwimmbad import MPIPool
    #from schwimmbad import MultiPool


    # generate mock cov
    if 'generate_mockcov' in params:
        if params['generate_mockcov'] :
            generate_mock_covariance( params )
	    sys.exit(-1)
        else : pass
    else : pass


    # MPI for mock fitting
    if 'fitting_mocks' in params:
        if params['fitting_mocks']:
            pool = MPIPool()
            #pool = None
            #mock_pararrel_processing(params, pool=pool)
	    mock_pararrel_processing2(params, pool=pool)
            pool.close()

        else : 
            pool = None
            run_mcmc(params, pool=pool)
            #pool.close()


        """
        if 'fitting_mocks' in params:
            if params['fitting_mocks']:

                n_mocks = params['n_mocks']
                mock_dir = params['mock_dir']
                mock_probe = params['mock_probe']
                #if 'p' in params['probe'] : probe = 'p'
                #if 'xi' in params['probe'] : probe = 'xi'

                for i in range( 1, n_mocks+1 ):
                    
                    filename = mock_dir + 'mock_pxi_no{:04d}.txt'.format(i)
                    print 'calling mock datav :', filename
                    params['datav_filename'] = filename
                    params['name'] = 'mock_'+mock_probe+'_no{:04d}'.format(i)

                    pool = MPIPool()
                    #pool = None
                    run_mcmc(params, pool=pool)
                    pool.close()
            else : 
                pool = None
                run_mcmc(params, pool=pool)
                pool.close()
        """

    else : 
        #from schwimmbad import MultiPool
        #pool = MPIPool()
        pool = None
        run_mcmc(params, pool=pool)
        if pool is not None : pool.close()

