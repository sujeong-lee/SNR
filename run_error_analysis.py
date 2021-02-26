import time, datetime
import sys, os
import fitsio
import numpy as np
from numpy import zeros, sqrt, pi, vectorize
from numpy.linalg import pinv, inv
from multiprocessing import Process, Queue
#import matplotlib
#matplotlib.use('Qt4Agg')
#import matplotlib.pyplot as plt
#from error_analysis_class import *
sys.path.append('src/')
#from noshellavg_v2 import *
from discrete import *
#from matplotlib.backends.backend_pdf import PdfPages

import argparse
import yaml
    
def run_error_analysis(params):
 
    kmin, kmax, kN = params['k']
    rmin, rmax, rN = params['r']
    #logscale = params['logscale']
    KMIN, KMAX = 1e-04, 10.
    if 'KRANGE_Fourier' in params : 
        KMIN, KMAX = params['KRANGE_Fourier']

    lmax = params['lmax']
    parameter_ind = params['parameter_ind']  
    kscale = 'log'
    if 'kscale' in params : kscale = params['kscale']
    rscale = 'lin'
    if 'rscale' in params : rscale = params['rscale']
    #parameter_ind_xi = params['parameter_ind_xi'] 

    b = 2.0
    if 'b' in params: b = params['b']
    f = 0.74
    if 'f' in params: f = params['f']
    s = 3.5
    if 's' in params: s = params['s']
    nn = 3.0e-04
    if 'nn' in params: nn = params['nn']
    parameter_names = np.array(['b', 'f', 's', 'nn'])
    
    params['parameter'] = str(parameter_names[parameter_ind]) 

    #print '-----------------------------------'
    #print ' Run Error Analaysis'
    #print '-----------------------------------'
    print ' parameter setting'
    print ' b={} f={} s={} nn={}'.format(b,f,s,nn)
    print ' free params :'+ str(parameter_names[parameter_ind]) 
    print ' k = [{}, {}], kN={}'.format(kmin, kmax, kN)
    print ' r = [{}, {}], rN={}'.format(rmin, rmax, rN)
    print ' lmax={}'.format(lmax)
    print ' kscale:', kscale, ', rscale:', rscale
    print '-----------------------------------'
    
    #RSDPower = NoShell_covariance(KMIN, KMAX, rmin, rmax, 2**12 + 1, rN, kN, b,f,s,nn,logscale = logscale)
    RSDPower = class_discrete_covariance(KMIN=KMIN, KMAX=KMAX, RMIN=rmin, RMAX=rmax, n=20000, n_y = kN, n2=rN, b=b, f=f,
     s=s, nn=nn, kscale = kscale, rscale=rscale)
    
    if 'matterpower' in params :
        file = params['matterpower']
    else : 
        file = 'src/matterpower_z_0.55.dat'  # from camb (z=0.55)

    #lik_class.mPk_file = file
    RSDPower.mPk_file = file #'src/matterpower_z_0.55.dat'
    print 'calling stored matter power spectrum.. ', file

    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    params['kmincut'] = RSDPower.kmin_y[kcut_min]
    params['kmaxcut'] = RSDPower.kmax_y[kcut_max]
    print 'kmincut=', params['kmincut'], ' kmaxcut=',params['kmaxcut']
    #params['kbin'] = RSDPower.kbin_y
    #params['kcenter'] = RSDPower.kcenter_y
    #params['rbin'] = RSDPower.rbin
    #params['rcenter'] = RSDPower.rcenter

    Covariance_matrix(params, RSDPower)
    Calculate_Fisher_tot(params, RSDPower, kmin = kmin, kmax = kmax, lmax=lmax)
    
    np.savetxt(params['savedir']+'kbin.txt', RSDPower.kbin_y)
    np.savetxt(params['savedir']+'kcenter.txt', RSDPower.kcenter_y)
    np.savetxt(params['savedir']+'rbin.txt', RSDPower.rbin)
    np.savetxt(params['savedir']+'rcenter.txt', RSDPower.rcenter)

    """
    print '\nStore Bessel functions'
    from fortranfunction import sbess
    sbess = np.vectorize(sbess)
    m1, m2 = np.mgrid[0:RSDPower.kcenter.size, 0:RSDPower.rcenter.size]
    
    sbess_vector = sbess(0, RSDPower.kcenter[m1]*RSDPower.rcenter[m2])
    np.savetxt(params['savedir']_sbess0.txt',sbess_vector)
    
    Vi = 4./3 * np.pi * (RSDPower.rmax**3 - RSDPower.rmin**3)
    AvgBessel = avgBessel(0, RSDPower.kcenter[m1], RSDPower.rmin[m2], RSDPower.rmax[m2] )/Vi[m2]
    np.savetxt(params['savedir']_avgbessel0.txt',AvgBessel)
    """
                      
    
    if 'multipole_P_filename' not in params:
        P_multipole(params, RSDPower)
        #np.savetxt(params['savedir']+'multipole_p.datavector',multipole_datav )
        #params['multipole_P_filename'] = params['savedir']+'multipole_p.datavector'
    else : print '\nUse Precalculated multipole_p ', params['savedir']+params['multipole_P_filename']
        
    if 'derivative_P_filename' not in params:
        derivative_P_datavector(params, RSDPower)
    else : print 'Use Precalculated derivative_P ', params['derivative_P_filename']     
        
        
    if 'derivative_Xi_filename' not in params:
        derivative_Xi_datavector(params, RSDPower)
    else : print 'Use Precalculated derivative_Xi ', params['derivative_Xi_filename']    
      
    if 'params_datavector_filename' not in params:
        params_datavector(params, RSDPower)
    else : print '\nUse Precalculated params_datavector ', params['params_datavector_filename']     
        
    
    #BandpowerFisher(params, RSDPower, kmin = kmin, kmax = kmax, lmax = lmax) 
    #Fisher_params(params, RSDPower, parameter = parameter_ind, kmin=kmin, kmax=kmax, lmax=lmax)
    
    direct_projection = 0
    if 'direct_projection' in params:
        direct_projection = params['direct_projection']

    if direct_projection :
        if 'params_xi_datavector_filename' not in params:
            params_xi_datavector(params, RSDPower)
        else : 
            print '\nUse Precalculated params_si_datavector ', params['params_xi_datavector_filename'] 
        #DirectProjection_to_params(params, RSDPower, parameter =parameter_ind, kmin=kmin, kmax=kmax, lmax=lmax)
        DirectProjection_to_params_shotnoise(params, RSDPower, kmin = kmin, kmax = kmax, lmax =lmax, p_parameter = parameter_ind, xi_parameter =parameter_ind )
   
    else : 
        BandpowerFisher(params, RSDPower, kmin = kmin, kmax = kmax, lmax = lmax) 
        Fisher_params(params, RSDPower, parameter = parameter_ind, kmin=kmin, kmax=kmax, lmax=lmax)

    SNR = params['SNR']
    if SNR : 
        print '\n\ncalculating SNR...'
        CumulativeSNR(params, RSDPower, kmin=kmin, kmax=kmax, lmax=lmax)

    Reid = params['Reid']  
    if Reid : 
        print 'calclating Reid result...'
        Reid_error(params, RSDPower, parameter = parameter_ind, lmax=lmax)
    
    
    #fitsio.write( params['savedir']+'output.fits', params, clobber=True )
    

    
def Covariance_matrix(params, RSDPower):    
    
    save_dir = params['savedir']+'/'
    lmax = int(params['lmax'])

    file = None #'matterpower_z_0.55.dat'  # from camb (z=0.55)
    RSDPower.MatterPower(file = file)
    RSDPower.multipole_P_band_all()
    # P covariance matrix ( nine submatrices C_ll' )

    if 'covPP_filename' not in params:
        RSDPower.RSDband_covariance_PP_all(lmax=lmax)
        C_matrix3PP = np.vstack((
                np.hstack([RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04 ]),\
                np.hstack([RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24 ]),\
                np.hstack([RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44 ])
                ))
        f = save_dir+'P.cov'
        np.savetxt(f, C_matrix3PP)
        params['covPP_filename'] = f

    else : print 'Use Precalculated CovPP ', params['covPP_filename']
     
    
    if 'covXi_filename' not in params:
        RSDPower.covariance_Xi_all(lmax=lmax)
        C_matrix3Xi = np.vstack((
                np.hstack([RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04 ]),\
                np.hstack([RSDPower.covariance02.T, RSDPower.covariance22, RSDPower.covariance24 ]),\
                np.hstack([RSDPower.covariance04.T, RSDPower.covariance24.T, RSDPower.covariance44 ])
                ))
        #f2 = params['savedir']'Xi.cov'
        f2 = save_dir+'Xi.cov'
        np.savetxt(f2, C_matrix3Xi) 
        params['covXi_filename'] = f2

    else : print 'Use Precalculated CovXi ', params['covXi_filename']
        
    
    if 'covPXi_filename' not in params:
        RSDPower.covariance_PXi_All(lmax=lmax)
        C_matrix3PXi = np.vstack((
                np.hstack([RSDPower.covariance_PXi00, RSDPower.covariance_PXi02, RSDPower.covariance_PXi04 ]),\
                np.hstack([RSDPower.covariance_PXi20, RSDPower.covariance_PXi22, RSDPower.covariance_PXi24 ]),\
                np.hstack([RSDPower.covariance_PXi40, RSDPower.covariance_PXi42, RSDPower.covariance_PXi44 ])
                ))
        
        #f3 = params['savedir']'PXi.cov'
        f3 = save_dir+'PXi.cov'
        np.savetxt(f3, C_matrix3PXi)
        params['covPXi_filename'] = f3

    else : print 'Use Precalculated CovPXi ', params['covPXi_filename']

    ##### end #### -----------------------------------------------------------------------
    
def P_multipole(params, RSDPower):

    # power spectrum multipoles l = 0,2,4
    #RSDPower.multipole_P_band_all() 
    RSDPower.multipole_bandpower0 = RSDPower.multipole_P(0) + 1./RSDPower.nn
    RSDPower.multipole_bandpower2 = RSDPower.multipole_P(2)
    RSDPower.multipole_bandpower4 = RSDPower.multipole_P(4)
    
    multipole_datav = np.hstack([RSDPower.multipole_bandpower0,RSDPower.multipole_bandpower2\
                             ,RSDPower.multipole_bandpower4])
    #np.savetxt(params['savedir']'+'multipole_p.datavector',multipole_datav )
    #    params['multipole_P_filename'] = params['savedir']'+'multipole_p.datavector'

    if params is None : pass
    else : 
        f = params['savedir']+'multipole_p.datavector'
        np.savetxt(f,multipole_datav)
        params['multipole_P_filename'] = f

    return multipole_datav
    
def Xi_multipole(params, RSDPower):

    # power spectrum multipoles l = 0,2,4
    
    #RSDPower.multipole_Xi_all()
    RSDPower.multipole_xi0 = RSDPower.multipole_Xi(0)
    RSDPower.multipole_xi2 = RSDPower.multipole_Xi(2)
    RSDPower.multipole_xi4 = RSDPower.multipole_Xi(4)
    
    multipole_datav = np.hstack([RSDPower.multipole_xi0,RSDPower.multipole_xi2\
                             ,RSDPower.multipole_xi4])
    #np.savetxt(params['savedir']'+'multipole_xi.datavector',multipole_datav )
    #params['multipole_Xi_filename'] = params['savedir']'+'multipole_xi.datavector'
        
    if params is None : pass
    else : 
        f = params['savedir']+'multipole_xi.datavector'
        np.savetxt(f,multipole_datav)
        params['multipole_Xi_filename'] = f

    return multipole_datav
    
    
def derivative_P_datavector(params, RSDPower):    

    derivative_P0 = np.identity(RSDPower.kcenter_y.size)# [:,kcut_min:kcut_max+1]
    Pzeros = np.zeros((derivative_P0.shape))

    derivative_P = np.concatenate((np.concatenate((derivative_P0, Pzeros, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, derivative_P0, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, Pzeros, derivative_P0),axis=1 )), axis=0)
    #f = params['savedir']'+'P.datavector'
    f = params['savedir']+'DP_datavector.txt'
    np.savetxt(f,derivative_P)
    params['derivative_P_filename'] = f
    
        
def _derivative_Xi_datavector(params, RSDPower):        

    """
    make sum of dxi/dp forcefully 0 by adding extra terms.
    This make amplitude of snr_xi decrease.
    """
    RSDPower.derivative_Xi_band_all()
    Xizeros = np.zeros((RSDPower.dxip0.shape))
    
    print 'RSDPower.dxip0 shape', Xizeros.shape
    
    last_new0 = -1*np.sum(RSDPower.dxip0[:-1,:], axis = 0)
    RSDPower.dxip0[-1,:] = last_new0
    
    last_new2 = -1*np.sum(RSDPower.dxip2[:-1,:], axis = 0)
    RSDPower.dxip2[-1,:] = last_new2
    
    last_new4 = -1*np.sum(RSDPower.dxip4[:-1,:], axis = 0)
    RSDPower.dxip4[-1,:] = last_new4
    
    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0,Xizeros,Xizeros), axis=1),\
                                             np.concatenate((Xizeros,RSDPower.dxip2,Xizeros), axis=1),\
                                             np.concatenate((Xizeros,Xizeros,RSDPower.dxip4), axis=1)),axis=0 )

    #f2 = params['savedir']'+'Xi.datavector'
    f2 = params['savedir']+'DXi_datavector.txt'
    np.savetxt(f2,derivative_correl_avg)
    params['derivative_Xi_filename'] = f2

def derivative_Xi_datavector(params, RSDPower):        

    RSDPower.derivative_Xi_band_all()
    Xizeros = np.zeros((RSDPower.dxip0.shape))
    
    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0,Xizeros,Xizeros), axis=1),\
                                             np.concatenate((Xizeros,RSDPower.dxip2,Xizeros), axis=1),\
                                             np.concatenate((Xizeros,Xizeros,RSDPower.dxip4), axis=1)),axis=0 )

    #f2 = params['savedir']'+'Xi.datavector'
    f2 = params['savedir']+'DXi_datavector.txt'
    np.savetxt(f2,derivative_correl_avg)
    params['derivative_Xi_filename'] = f2
    
    
    
    
    ## end #####################################################################
    
    
def params_datavector(params, RSDPower):
    
    #if 'params_datavector_filename' not in params:
        
    # derivative dXidb, s, f, n
    #RSDPower.derivative_bfs_all()
    RSDPower.derivative_P_bfs_all()

    # add shot noise params
    dPN0 = np.ones(RSDPower.kcenter_y.size) * (-1./RSDPower.nn**2)
    dPN1 = np.zeros(RSDPower.kcenter_y.size)
    dPN2 = dPN1.copy()

    matrices2P = np.vstack((
            np.hstack([RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4]),\
            np.hstack([RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4]),\
            np.hstack([RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]),\
            np.hstack([dPN0, dPN1, dPN2]) ))

    
    #f = params['savedir']'+'params.datavector'
    f = params['savedir']+'DP_params_datavector.txt'
    np.savetxt(f, matrices2P)
    params['params_datavector_filename'] = f
    #else : print 'Use Precalculated params_datavector ', params['params_datavector_filename']

    ### end ####################################################################        
    

def params_xi_datavector(params, RSDPower):
    from fortranfunction import sbess
    #if 'params_datavector_filename' not in params:  
    # derivative dXidb, s, f, n
    RSDPower.derivative_bfs_all()
    #RSDPower.derivative_P_bfs_all()

    
    kmax = RSDPower.KMAX
    kmin = RSDPower.KMIN
    r = RSDPower.rcenter
    
    #dxin0 = 4.*np.pi*(-kmax*r*np.cos(kmax*r) + kmin*r*np.cos(kmin*r) + np.sin(kmax*r) -  np.sin(kmin*r))/(r**3)/(2*np.pi)**3
    #dxin0 = dxin0 * (-1./RSDPower.nn**2)
    dxin0 = np.zeros(RSDPower.rcenter.size)
    
    dxin2 = np.zeros(RSDPower.rcenter.size)
    dxin4 = dxin2.copy()
    
    
    matrices2P = np.vstack((
            np.hstack([RSDPower.dxib0, RSDPower.dxib2, RSDPower.dxib4]),\
            np.hstack([RSDPower.dxif0, RSDPower.dxif2, RSDPower.dxif4]),\
            np.hstack([RSDPower.dxis0, RSDPower.dxis2, RSDPower.dxis4]),\
            np.hstack([dxin0, dxin2, dxin4]) ))


    #f = params['savedir']'+'params_xi.datavector'
    f = params['savedir']+'DXi_params_datavector.txt'

    np.savetxt(f, matrices2P)
    params['params_xi_datavector_filename'] = f
    #else : print 'Use Precalculated params_datavector ', params['params_datavector_filename']

    ### end ####################################################################        

"""   
def masking(RSDPower, data, kmin = 0, kmax = 2, lmax = 4, xi=False):
    
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    if xi is True : kcut_min, kcut_max = 0, RSDPower.rcenter.size
    
    Nx, Ny = data.shape
    
    if Nx == Ny : 
        
        mask0_x, mask0_y = np.zeros((Nx/3, Nx/3)), np.zeros((Nx/3, Nx/3))
        mask0_x[:,kcut_min:kcut_max+1] = 1
        mask0_y[kcut_min:kcut_max+1,:] = 1
        mask0 = mask0_x*mask0_y

        mask1 = np.hstack([mask0, mask0, mask0])
        mask = np.vstack([mask1, mask1, mask1])
    
    elif Nx != Ny :    
        mask0 = np.zeros((RSDPower.kcenter_y.size, Ny/3))
        mask0[kcut_min:kcut_max+1, :] = 1
        mask1 = np.hstack((mask0, mask0, mask0))
        mask = np.vstack([mask1, mask1, mask1])
              
    if lmax == 0:
        mask2= np.zeros((Nx, Ny))
        mask2[:Nx/3,:Ny/3] = 1
        mask = mask * mask2

    elif lmax == 2:
        mask4= np.zeros((Nx, Ny))
        mask4[:2*Nx/3,:2*Ny/3] = 1
        mask = mask * mask4       
        
    return data * mask
"""      
 
    
"""
def masking_datav(RSDPower, data, kmin = 0, kmax = 2, lmax = 4, xi=False):
    
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    if xi : 
        kcut_min, kcut_max = 0, RSDPower.rcenter.size
        
    Nx, Ny = data.shape
  
    mask1 = np.zeros((Nx, Ny/3))
    mask1[:,kcut_min:kcut_max+1] = 1
    mask = np.hstack([mask1, mask1, mask1])
    
    
    if lmax == 0:
        mask2= np.zeros((Nx, Ny))
        mask2[:Nx/3,:Ny/3] = 1
        mask = mask * mask2

    elif lmax == 2:
        mask4= np.zeros((Nx, Ny))
        mask4[:2*Nx/3,:2*Ny/3] = 1
        mask = mask * mask4
        
    return data * mask
    
"""


    
def masking(RSDPower, data, kmin = None, kmax = None, lmax = 4, xi=False, pxi=False):
    
    if xi is True : kcut_min, kcut_max = 0, RSDPower.rcenter.size
    else :
        kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
        kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    Nx, Ny = data.shape

    #if Nx%3 !=0 : stop
    
    mask2= np.zeros((Nx, Ny), dtype=bool)
    if lmax == 0: l = 1
    elif lmax == 2: l = 2
    else : l = 3

    mask2[:l*Nx/3,:l*Ny/3] = 1

    if pxi:  
    #elif Nx != Ny :  
        mask0 = np.zeros((Nx/3, Ny/3), dtype=bool)
        #print 'mask0', mask0.shape

        mask0[kcut_min:kcut_max+1, :] = 1
        #if lmax == 0 : mask = mask0 * mask2
        #elif lmax == 2 :
        #    mask1 = np.hstack((mask0, mask0))
        #    mask = np.vstack([mask1, mask1]) * mask2           
        #elif lmax == 4 :
        mask1 = np.hstack((mask0, mask0, mask0))
        mask = np.vstack([mask1, mask1, mask1]) * mask2

        

        data = data[mask]
        ny = Ny/3 * l
        nx = data.size/ny

    #if Nx == Ny : 
    else:    
        mask0_x, mask0_y = np.zeros((Nx/3, Nx/3), dtype=bool), np.zeros((Nx/3, Nx/3), dtype=bool)
        mask0_x[:,kcut_min:kcut_max+1] = 1
        mask0_y[kcut_min:kcut_max+1,:] = 1
        mask0 = mask0_x*mask0_y

        mask1 = np.hstack([mask0, mask0, mask0])
        #print mask1.shape, mask2.shape
        #print mask0.shape, mask1.shape, mask2.shape
        #print mask2.shape, mask1.shape 
        
        #if lmax == 0 : mask = mask1 * mask2
        #elif lmax == 2 : mask = np.vstack([mask1, mask1]) * mask2
        #elif lmax == 4: 
        mask = np.vstack([mask1, mask1, mask1]) * mask2
        data = data[mask]
        nx = int(np.sqrt(data.size))
        ny = nx
        
    return data.reshape(nx, ny)


def masking_datav(RSDPower, data, kmin = None, kmax = None, lmax = 4, xi=False):
    
    if xi : kcut_min, kcut_max = 0, RSDPower.rcenter.size
    else :
        kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
        kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    Nx, Ny = data.shape

    mask2= np.zeros((Nx, Ny), dtype=bool)
    if lmax == 0: l = 1
    elif lmax == 2: l = 2
    else : l = 3
    mask2[:l*Nx/3,:l*Ny/3] = 1

    
    mask1 = np.zeros((Nx, Ny/3), dtype=bool)    
    mask1[:,kcut_min:kcut_max+1] = 1
    mask = np.hstack([mask1, mask1, mask1]) * mask2    
    
    data = data[mask]
    nx = l*Nx/3
    ny = data.size/nx
        
    return data.reshape(nx, ny)


def generate_mask_datav(RSDPower, data, kmin = None, kmax = None, lmax = 4, xi=False):
    
    if xi : kcut_min, kcut_max = 0, RSDPower.rcenter.size
    else :
        kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
        kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    if len(data.shape) == 1 : data = data.reshape(1, data.size)    
    Nx, Ny = data.shape
  
    mask2= np.zeros((Nx, Ny), dtype=bool)
    if lmax == 0: l = 1
    elif lmax == 2: l = 2
    elif lmax == 4: l = 3
        
    mask2[:,:l*Ny/3] = 1
    
    mask1 = np.zeros((Nx, Ny/3), dtype=bool)
    mask1[:,kcut_min:kcut_max+1] = 1
    mask = np.hstack([mask1, mask1, mask1]) * mask2
    return mask


def masking_paramsdatav(RSDPower, data, kmin = None, kmax = None, lmax = 4, xi=False):
    
    if xi : 
        kcut_min, kcut_max = 0, RSDPower.rcenter.size
    else : 
        kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )   
        kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )

    if len(data.shape) == 1 : data = data.reshape(1, data.size)
    Nx, Ny = data.shape
    # Ny : k or r direction
    
    mask2= np.zeros((Nx, Ny), dtype=bool)
    if lmax == 0: l = 1
    elif lmax == 2: l = 2
    elif lmax == 4: l = 3
        
    mask2[:,:l*Ny/3] = 1

    mask1 = np.zeros((Nx, Ny/3), dtype=bool)
    mask1[:,kcut_min:kcut_max+1] = 1
    mask = np.hstack([mask1, mask1, mask1]) * mask2
    
    data = data[mask]
    ny = data.size/Nx
    #print data.shape, ny, Nx
    return data.reshape(Nx, ny)
    

    
def BandpowerFisher(params, RSDPower, kmin = 0, kmax = 10, lmax=4):
    
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    covPP_masked = masking(RSDPower, covPP, kmin = kmin, kmax = kmax, lmax=lmax)
    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax, pxi=True)
    #print 'covariance matrix size', covPP_masked.shape, covXi.shape, covPXi.shape 
    
    
    #C_tot = np.concatenate((np.concatenate((covPP_masked, covPXi), axis=1),
    #                        np.concatenate((covPXi.T, covXi), axis=1)), axis = 0)

    datav_P = masking_datav(RSDPower, np.genfromtxt(params['derivative_P_filename']), kmin = kmin, kmax = kmax, lmax=lmax)
    datav_Xi = masking_datav(RSDPower, np.genfromtxt(params['derivative_Xi_filename']), xi=True, lmax=lmax)   
    
    datav = np.concatenate((datav_P, datav_Xi), axis=1)    
    # inverting matrices
    #from test_SNR import blockwiseInversion

    
    if 'fisher_bandpower_P_filename' not in params:               
        #FisherP = pinv(covPP)
        
        if lmax == 0 : 
            FisherP = np.zeros(covPP_masked.shape) 
            np.fill_diagonal(FisherP, 1./covPP_masked.diagonal())
        elif lmax == 2 : 
            #raise ValueError('two modes not implemented yet')
            FisherP = inv(covPP_masked)
            
        elif lmax == 4 : 
            
            cut = RSDPower.kcenter_y.size
            covPPlist = [covPP[:cut, :cut], covPP[:cut, cut:2*cut], covPP[:cut, 2*cut:],
                        covPP[:cut, cut:2*cut], covPP[cut:2*cut, cut:2*cut], covPP[cut:2*cut, 2*cut:], 
                        covPP[:cut, 2*cut:], covPP[cut:2*cut, 2*cut:], covPP[2*cut:, 2*cut:]]
            FisherP = masking(RSDPower, DiagonalBlockwiseInversion3x3(*tuple(covPPlist)), kmin=kmin, kmax=kmax, lmax=lmax)
            #FisherP = pinv( covPP_masked, rcond=1e-15)
            #print 'Diagonal blockwise inversion not used'
            #FisherBand_P = FisherProjection_Fishergiven(datav_P, FisherP)
            #FisherBand_P = np.dot(np.dot(datav_P, FisherP), datav_P.T)
        if np.sum(FisherP.diagonal()<=0) != 0 : raise ValueError('Inversion Failed')    
        #f = params['savedir']'bandpower_PP.fisher'
        f = params['savedir']+'P_fisher.txt'
        np.savetxt(f, FisherP)
        params['fisher_bandpower_P_filename']= f
        print '\nFisherP saved ', f
        
    else : 
        FisherP = np.genfromtxt(params['fisher_bandpower_P_filename'])
        print '\nUse Precalculated FisherP ', params['fisher_bandpower_P_filename']
        
    if 'fisher_bandpower_Xi_filename' not in params:
        FisherXi = pinv(covXi, rcond=1e-15)
        if np.sum(FisherXi.diagonal()<=0) != 0 : raise ValueError('Inversion Failed')
            
        #f2 = params['savedir']'Xi.fisher'
        f2 = params['savedir']+'Xi_fisher.txt'
        np.savetxt(f2, FisherXi)
        params['fisherXi_filename']= f2  
        print 'FisherXi saved ', f2

        #print '\nFisherXi', np.sum(FisherXi.diagonal() <= 0.0)
        FisherBand_Xi = FisherProjection_Fishergiven(datav_Xi, FisherXi)
        if np.sum(FisherBand_Xi.diagonal()<=0) != 0 : raise ValueError('Inversion Failed')
        #FisherBand_Xi = np.dot(np.dot(datav_Xi, FisherXi), datav_Xi.T)
        #f2 = params['savedir']'bandpower_Xi.fisher'
        f2 = params['savedir']+'bandpowerXi_fisher.txt'
        np.savetxt(f2, FisherBand_Xi)
        params['fisher_bandpower_Xi_filename']= f2   
        print 'Fisher_bandpower_Xi saved ', f2
        
    else : 
        FisherXi = np.genfromtxt(params['fisher_bandpower_Xi_filename'])
        print 'Use Precalculated FisherXi ', params['fisher_bandpower_Xi_filename']
        

    if 'com' in params['probe']:

        if 'fisher_bandpower_tot_filename' not in params:    
        
            print 'calculating Fisher tot, Blockwise Inversion'

            b = covPXi
            c = covPXi.T #matrix[cutInd+1:, 0:cutInd+1]
            d = covXi
            ia = FisherP #masking(RSDPower, FisherP, kmin=kmin, kmax=kmax)
            
            Fd = pinv( d - np.dot( np.dot( c, ia ), b) )
            Fc = - np.dot( np.dot( Fd, c), ia)
            Fb = - np.dot( np.dot( ia, b ), Fd )
            Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)

            Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
            print 'Fisher3_tot diagonal test', np.sum(Fisher3_tot.diagonal() < 0)
            ## if np.sum(Fisher3_tot.diagonal()<0) != 0 : raise ValueError('Inversion Failed')


            
            #Fisher3_tot = pinv(C_tot, rcond = 1e-30)
            #FisherBand_tot = np.dot( np.dot( datav, Fisher3_tot), datav.T)
            FisherBand_tot = FisherProjection_Fishergiven(datav, Fisher3_tot)
            #print 'FisherBand_tot diagonal test', np.sum(FisherBand_tot.diagonal() <0)
            ## if np.sum(FisherBand_tot.diagonal()<0) != 0 : raise ValueError('Inversion Failed')

            #f3 = params['savedir']'bandpower_tot.fisher'
            f3 = params['savedir']+'bandpower_tot_fisher.txt'
            np.savetxt(f3, FisherBand_tot)
            params['fisher_bandpower_tot_filename']= f3
            print 'Fishertot saved ', f3
        else : 
            FisherBand_tot = np.genfromtxt( params['fisher_bandpower_tot_filename'])
            print 'Use Precalculated Fisher_tot ', params['fisher_bandpower_tot_filename']

    ##### end #########################################


    #Diagonal matrix inversion

def DiagonalBlockwiseInversion3x3( mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9 ):
    
    a = mat1.diagonal()
    b = mat2.diagonal()
    c = mat3.diagonal()
    d = mat4.diagonal()
    e = mat5.diagonal()
    f = mat6.diagonal()
    g = mat7.diagonal()
    h = mat8.diagonal()
    i = mat9.diagonal()

    A = np.zeros((a.size, a.size))
    B = np.zeros((a.size, a.size))
    C = np.zeros((a.size, a.size))
    D = np.zeros((a.size, a.size))
    E = np.zeros((a.size, a.size))
    F = np.zeros((a.size, a.size))
    G = np.zeros((a.size, a.size))
    H = np.zeros((a.size, a.size))
    I = np.zeros((a.size, a.size))

    a_s = 1./(a-b*d/e)
    b_s = - a_s *b/e
    c_s = - d/e*a_s
    d_s = 1./e + d/e * a_s * b/e

    F_22 = (i - (g*a_s*c + h*c_s*c + g*b_s*f + h*d_s*f ))**(-1)

    F_12_1 = - F_22 * (a_s * c + b_s * f)
    F_12_2 = - F_22 * (c_s * c + d_s * f)

    F_21_1 = - F_22 * (g*a_s + h*c_s)
    F_21_2 = - F_22 * (g*b_s + h*d_s)


    F_11_11 = a_s * (1 + F_22 * (c*g*a_s + c*h*c_s) ) + b_s * F_22 * (f*g*a_s + f*h*c_s)
    F_11_12 = a_s * F_22 * (c*g*b_s + c*h*d_s) + b_s * (1+F_22*(f*g*b_s + f*h*d_s))
    F_11_21 = c_s * (1 + F_22 * (c*g*a_s + c*h*c_s) ) + d_s * F_22 * (f*g*a_s + f*h*c_s)
    F_11_22 = c_s * F_22 * (c*g*b_s + c*h*d_s) + d_s * (1+F_22*(f*g*b_s + f*h*d_s))


    np.fill_diagonal(A, F_11_11 )
    np.fill_diagonal(B, F_11_12 )
    np.fill_diagonal(D, F_11_21 )
    np.fill_diagonal(E, F_11_22 )

    np.fill_diagonal(C, F_12_1 )
    np.fill_diagonal(F, F_12_2 )

    np.fill_diagonal(G, F_21_1 )
    np.fill_diagonal(H, F_21_2 )

    np.fill_diagonal(I, F_22 )
    
    FF = np.vstack((np.hstack([A,B,C]), np.hstack([D,E,F]), np.hstack([G,H,I]) ))
    
    return FF

def DirectProjection_to_params(params, RSDPower, parameter =[0,1,2,3], kmin = 0, kmax = 10, lmax = 4, diffs = False):
    
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    covPP_masked = masking(RSDPower, covPP, kmin=kmin, kmax=kmax, lmax=lmax)
    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax)
    
    #covPP_masked = masking(RSDPower, covPP, kmin = kmin, kmax = kmax)
    #C_tot = np.concatenate((np.concatenate((covPP_masked, covPXi), axis=1),
    #                        np.concatenate((covPXi.T, covXi), axis=1)), axis = 0)
    
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    params_datav_mar = np.vstack(([ params_datav[p,:] for p in parameter] ))
    params_datav_mar_kcut = masking_paramsdatav(RSDPower, params_datav_mar, kmin=kmin, kmax=kmax, lmax=lmax)
    
    
    params_xi_datav = np.genfromtxt(params['params_xi_datavector_filename'])
    params_xi_datav_mar = np.vstack(([ params_xi_datav[p,:] for p in parameter] ))
    params_xi_datav_mar = masking_paramsdatav(RSDPower, params_xi_datav_mar, xi=True, lmax=lmax)

    if diffs : 
        dpss = np.zeros(params_datav_mar_kcut.shape[1])
        params_datav_mar_kcut = np.insert(params_datav_mar_kcut, 3, dpss, axis=0 )
        
        dxiss = np.zeros(params_xi_datav_mar.shape[1])
        params_xi_datav_mar = np.insert(params_xi_datav_mar, 2, dxiss, axis=0 )
    
    datav = np.concatenate((params_datav_mar_kcut,params_xi_datav_mar), axis=1)
    
    
    # inverting matrices
    from test_SNR import blockwiseInversion
    
    if 'fisher_bandpower_P_filename' not in params: 
        
        if lmax == 0 : 
            FisherP = np.zeros(covPP_masked.shape) 
            np.fill_diagonal(FisherP, 1./covPP_masked.diagonal())
        elif lmax == 2 : raise ValueError('two modes not implemented yet')
        elif lmax == 4 : 
            
            cut = RSDPower.kcenter_y.size
            covPPlist = [covPP[:cut, :cut], covPP[:cut, cut:2*cut], covPP[:cut, 2*cut:],
                        covPP[:cut, cut:2*cut], covPP[cut:2*cut, cut:2*cut], covPP[cut:2*cut, 2*cut:], 
                        covPP[:cut, 2*cut:], covPP[cut:2*cut, 2*cut:], covPP[2*cut:, 2*cut:]]
            FisherP = masking(RSDPower, DiagonalBlockwiseInversion3x3(*tuple(covPPlist)), kmin=kmin, kmax=kmax, lmax=lmax)
       
    #else : FisherP = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_P_filename']), kmin=kmin, kmax=kmax, lmax=lmax)
    else : FisherP = np.genfromtxt(params['fisher_bandpower_P_filename'])
   
    print 'FisherP', FisherP.shape
    F_params_P = np.dot(np.dot(params_datav_mar_kcut, FisherP), params_datav_mar_kcut.T)
    
    
    
    if 'fisherXi_filename' not in params: FisherXi = pinv(covXi, rcond=1e-15)
    else : FisherXi = np.genfromtxt(params['fisherXi_filename'])
    if np.sum(FisherXi.diagonal() < 0) != 0 : 
        raise ValueError(' Inversion Failed! ')
        
    F_params_Xi = np.dot(np.dot(params_xi_datav_mar, FisherXi), params_xi_datav_mar.T)
        
    if 'fishertot_filename' not in params:    

        print 'calculating Fisher tot. Blockwise inversion'
        
        b = covPXi
        c = covPXi.T #matrix[cutInd+1:, 0:cutInd+1]
        d = covXi
        ia = FisherP#masking(RSDPower, FisherP, kmin=kmin, kmax=kmax)

        Fd = pinv( d - np.dot( np.dot( c, ia ), b) )
        Fc = - np.dot( np.dot( Fd, c), ia)
        Fb = - np.dot( np.dot( ia, b ), Fd )
        Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)

        Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
        
        
        #Fisher3_tot = pinv(C_tot, rcond = 1e-30)
        """
        rcondnum = np.arange(30, 13, -1)
        for rc in rcondnum:
            Fisher3_tot = pinv(C_tot, rcond = 10**(-1*rc))
            neg = np.sum(Fisher3_tot.diagonal() <=0.0)
            print rc, neg
            if neg == 0 : break
        if rc == 14 : raise ValueError("Inversion failed : rcond exceeds 1e-15")   
        """
    #else : Fisher3_tot = np.genfromtxt(params['fishertot_filename'])
    else : Fisher3_tot = np.dot(np.dot(datav, Fisher3_tot),datav.T)    
    F_params_tot = np.dot(np.dot(datav,Fisher3_tot), datav.T)

    ind = np.arange(0,(len(parameter))**2)
    if diffs == True : ind = np.arange(0,(len(parameter)+1)**2)
    DAT = np.column_stack((ind, F_params_P.ravel(), F_params_Xi.ravel(), F_params_tot.ravel() ))
    #f = params['savedir']+'fisher_params_direct.txt'
    f = params['savedir']+'fisher_params_direct.txt'
    np.savetxt(f, DAT)
    print 'save to', f



    


    
def Calculate_Fisher_tot(params, RSDPower, kmin = None, kmax = None, lmax = 4):
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    #print 'covp', covPP.shape
    covPP_masked = masking(RSDPower, covPP, kmin=kmin, kmax=kmax, lmax=lmax)

    #print 'covp masked', covPP_masked.shape

    #covXi_ = np.genfromtxt(params['covXi_filename'])
    #print 'covxi', covXi_.shape

    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    
    #covPXi_ = np.genfromtxt(params['covPXi_filename'])
    #print 'covxi', covXi.shape

    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax, pxi=True)

    #print 'covpxi', covPXi.shape


    if 'fisher_bandpower_P_filename' not in params: 
        
        if lmax == 0 : 
            FisherP = np.zeros(covPP_masked.shape) 
            np.fill_diagonal(FisherP, 1./covPP_masked.diagonal())
        elif lmax == 2 : 
            FisherP = inv(covPP_masked)
        elif lmax == 4 :  
            cut = RSDPower.kcenter_y.size
            covPPlist = [covPP[:cut, :cut],      covPP[:cut, cut:2*cut],      covPP[:cut, 2*cut:],
                         covPP[:cut, cut:2*cut], covPP[cut:2*cut, cut:2*cut], covPP[cut:2*cut, 2*cut:], 
                         covPP[:cut, 2*cut:],    covPP[cut:2*cut, 2*cut:],    covPP[2*cut:, 2*cut:]]
            FisherP = masking(RSDPower, DiagonalBlockwiseInversion3x3(*tuple(covPPlist)), kmin=kmin, kmax=kmax, lmax=lmax)
        else : ValueError('l should be 0, 2 or 4')
      
        #f = params['savedir']'bandpower_PP.fisher'
        f = params['savedir']+'bandpowerP_fisher.txt'
        np.savetxt(f, FisherP)
        params['fisher_bandpower_P_filename']= f
        print '\nFisherP saved ', f

    else : 
        FisherP = np.genfromtxt(params['fisher_bandpower_P_filename'])
        print 'Use Precalculated fisherP'
    
    if 'fisherXi_filename' not in params: 
        FisherXi = pinv(covXi, rcond=1e-15)
        if np.sum(FisherXi.diagonal() < 0) != 0: raise ValueError('Inversion Failed')
        #f2 = params['savedir']'Xi.fisher'
        f2 = params['savedir']+'Xi_fisher.txt'
        np.savetxt(f2, FisherXi)
        params['fisherXi_filename']= f2  
        print 'FisherXi saved ', f2
        
    else : 
        FisherXi = np.genfromtxt(params['fisherXi_filename'])
        print 'Use Precalculated fisherXi'

    
    if 'com' in params['probe']:
        if 'fishertot_filename' not in params:    
            
            print 'shotnoise :calculating Fisher tot, Blockwise Inversion'
            
            b = covPXi
            c = covPXi.T #matrix[cutInd+1:, 0:cutInd+1]
            d = covXi
            ia = FisherP #masking(RSDPower, FisherP, kmin=kmin, kmax=kmax, lmax=lmax)

            Fd = pinv( d - np.dot( np.dot( c, ia ), b), rcond=1e-15 )
            Fc = - np.dot( np.dot( Fd, c), ia)
            Fb = - np.dot( np.dot( ia, b ), Fd )
            Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)

            Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
            print 'Fisher3_tot diagonal check ',  np.sum(Fisher3_tot.diagonal() < 0), np.sum(Fd.diagonal() < 0), np.sum(Fa.diagonal() < 0)
            if np.sum(Fisher3_tot.diagonal() < 0) != 0: raise ValueError('Inversion Failed')
            f3 = params['savedir']+'fishertot.fisher' 
            params['fishertot_filename']= f3 
            np.savetxt(f3, Fisher3_tot)
            print 'Fisher3_tot saved', f3
        else : 
            Fisher3_tot = np.genfromtxt(params['fishertot_filename'])
            print 'Use Precalculated fishertot'
        
        
    
def DirectProjection_to_params_shotnoise(params, RSDPower, p_parameter =[0,1,2,3], xi_parameter =[0,1,2,3], kmin = None, kmax = None, lmax = 4):
    
    print '\nDirect Projection\n'
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    covPP_masked = masking(RSDPower, covPP, kmin=kmin, kmax=kmax, lmax=lmax)
    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax, pxi=True)
    
    #covPP_masked = masking(RSDPower, covPP, kmin = kmin, kmax = kmax)
    #C_tot = np.concatenate((np.concatenate((covPP_masked, covPXi), axis=1),
    #                        np.concatenate((covPXi.T, covXi), axis=1)), axis = 0)
    
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    
    params_datav_mar = np.vstack(([ params_datav[p,:] for p in p_parameter] ))
    params_datav_mar_kcut = masking_paramsdatav(RSDPower, params_datav_mar, kmin=kmin, kmax=kmax, lmax=lmax)
      
    params_xi_datav = np.genfromtxt(params['params_xi_datavector_filename'])
    params_xi_datav_mar = np.vstack(([ params_xi_datav[p,:] for p in xi_parameter] ))
    params_xi_datav_mar = masking_paramsdatav(RSDPower, params_xi_datav_mar, xi=True, lmax=lmax)
    #if 3 in xi_parameter : params_xi_datav_mar[-1,:] = 0
    #if len(xi_parameter) == 4: params_xi_datav_mar[-1,:] = 0   
    
    
    #if dinns : 
    #    dpnn = np.zeros(params_datav_mar_kcut.shape[1])
    #    params_datav_mar_kcut = np.insert(params_datav_mar_kcut, 4, dpnn, axis=0 )
    #    
    #    dxinn = np.zeros(params_xi_datav_mar.shape[1])
    #    params_xi_datav_mar = np.insert(params_xi_datav_mar, 3, dxinn, axis=0 )
        
        
    datav = np.concatenate((params_datav_mar_kcut,params_xi_datav_mar), axis=1)
    

    # inverting matrices
    #from test_SNR import blockwiseInversion

    if 'fisher_bandpower_P_filename' not in params: 
        
        if lmax == 0 : 
            FisherP = np.zeros(covPP_masked.shape) 
            np.fill_diagonal(FisherP, 1./covPP_masked.diagonal())
        elif lmax == 2 : 
            FisherP = inv(covPP_masked)
        elif lmax == 4 :  
            cut = RSDPower.kcenter_y.size
            covPPlist = [covPP[:cut, :cut], covPP[:cut, cut:2*cut], covPP[:cut, 2*cut:],
                        covPP[:cut, cut:2*cut], covPP[cut:2*cut, cut:2*cut], covPP[cut:2*cut, 2*cut:], 
                        covPP[:cut, 2*cut:], covPP[cut:2*cut, 2*cut:], covPP[2*cut:, 2*cut:]]
            FisherP = masking(RSDPower, DiagonalBlockwiseInversion3x3(*tuple(covPPlist)), kmin=kmin, kmax=kmax, lmax=lmax)
        else : ValueError('l should be 0, 2 or 4')
      
        f = params['savedir']+'bandpower_PP.fisher'
        np.savetxt(f, FisherP)
        params['fisher_bandpower_P_filename']= f
        print 'FisherP saved ', f

    else : 
        FisherP = np.genfromtxt(params['fisher_bandpower_P_filename'])
        print 'Use Precalculated fisherP'
                             
    F_params_P = np.dot(np.dot(params_datav_mar_kcut, FisherP), params_datav_mar_kcut.T)
    #if dinns : F_params_P[4][4] = 1e-20

    #print 'F-params_p\n', F_params_P
    
    if 'fisherXi_filename' not in params: 
        FisherXi = pinv(covXi, rcond=1e-15)
        if np.sum(FisherXi.diagonal() < 0) != 0: raise ValueError('Inversion Failed')
        f2 = params['savedir']+'Xi.fisher'
        np.savetxt(f2, FisherXi)
        params['fisherXi_filename']= f2  
        
        
    else : 
        FisherXi = np.genfromtxt(params['fisherXi_filename'])
        print 'Use Precalculated fisherXi'
    F_params_Xi = np.dot(np.dot(params_xi_datav_mar, FisherXi), params_xi_datav_mar.T)
    if len(xi_parameter) == 4: F_params_Xi[-1,-1] = 1e-100
    #print 'F_params_Xi[-1,-1] = ', F_params_Xi[-1,-1]
    #print 'F_params_Xi\n', F_params_Xi
    

    if 'com' in params['probe']:
        if 'fishertot_filename' not in params:    
            
            print 'shotnoise :calculating Fisher tot, Blockwise Inversion'
            
            b = covPXi
            c = covPXi.T #matrix[cutInd+1:, 0:cutInd+1]
            d = covXi
            ia = FisherP#masking(RSDPower, FisherP, kmin=kmin, kmax=kmax, lmax=lmax)

            Fd = pinv( d - np.dot( np.dot( c, ia ), b) )
            Fc = - np.dot( np.dot( Fd, c), ia)
            Fb = - np.dot( np.dot( ia, b ), Fd )
            Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)

            Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
            if np.sum(Fisher3_tot.diagonal() < 0) != 0: raise ValueError('Inversion Failed')
            f3 = params['savedir']+'fishertot.fisher' 
            params['fishertot_filename']= f3 
            np.savetxt(f3, Fisher3_tot)
            #print 'Fisher3_tot saved', f3
        else : 
            Fisher3_tot = np.genfromtxt(params['fishertot_filename'])
            print 'Use Precalculated fishertot'
      
        F_params_tot = FisherProjection_Fishergiven(datav, Fisher3_tot)
    

    from numpy.linalg import pinv as inv
    sigP = np.sqrt(inv( F_params_P ).diagonal())
    sigXi = np.sqrt(inv( F_params_Xi ).diagonal())
    if 'com' in params['probe']: sigtot = np.sqrt(inv( F_params_tot ).diagonal())
    sigdiff = np.sqrt(inv( F_params_P + F_params_Xi ).diagonal())

    parameter_name = ['b', 'f', 's', 'nn']

    print '\nname  :', 
    for i in range(len(p_parameter)): print parameter_name[i],
    print '\nP  :',
    for i in range(len(p_parameter)): print sigP[i],
    print '\nXi :',
    for i in range(len(p_parameter)): print sigXi[i],
    if 'com' in params['probe']:
        print '\ncom:',
        for i in range(len(p_parameter)): print sigtot[i],
    print '\ndif:',
    for i in range(len(p_parameter)): print sigdiff[i],
    print ''

    #DAT = np.column_stack((ind, Fqa.ravel(), Fqd4.ravel(), F_params_tot_q.ravel(), F_params_tot.ravel() ))
    #np.savetxt(params['savedir']+'fisher_params_nn.txt', DAT)
    #print '\nsave to', params['savedir']+'fisher_params_nn.txt'
    
    params['fisher_params_p_direct'] = F_params_P
    params['fisher_params_Xi_direct'] = F_params_Xi

    params['cov_params_p_direct'] = inv(F_params_P)
    params['cov_params_Xi_direct'] = inv(F_params_Xi)
    params['cov_params_diff_direct'] =  inv(F_params_P+F_params_Xi)

    params['sigma_params_p_direct'] = sigP 
    params['sigma_params_Xi_direct'] = sigXi
    params['sigma_params_diff_direct'] = sigdiff

    if 'com' in params['probe']:
        params['fisher_params_tot_direct'] = F_params_tot
        params['cov_params_tot_direct'] = inv(F_params_tot)
        params['sigma_params_tot_direct'] = sigtot 
        #fitsio.write( params['savedir']+'output.fits', params, clobber=True )
    

    ##### end #########################################



def CumulativeSNR(params, RSDPower, kmin=None, kmax=None, lmax=4):
    

    
    if lmax == 0 : l = 1
    elif lmax == 2 : l = 2
    elif lmax == 4 : l = 3
    
    from test_SNR import reorderingVector, reordering, _reordering, blockwise
    
    multipole_p = np.genfromtxt(params['multipole_P_filename'])
    multipole_p[:RSDPower.kcenter_y.size]-=1./RSDPower.nn

    datav_multipole = masking_paramsdatav(RSDPower, multipole_p.reshape(1, multipole_p.size)
                                    , kmin=RSDPower.KMIN, kmax = RSDPower.KMAX, lmax=lmax)     

    datav_multipole_kcut = masking_paramsdatav(RSDPower, multipole_p.reshape(1, multipole_p.size)
                                    , kmin=kmin, kmax=kmax, lmax=lmax)    
 
    datav_multipole_kcut_re = reorderingVector(datav_multipole_kcut, lmax=lmax)
    datav_multipole_re = reorderingVector(datav_multipole, lmax=lmax)
    
    ## loading fisher matrix
    #Fisher_P = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_P_filename']), 
    #                   kmin=RSDPower.KMIN, kmax=RSDPower.KMAX, lmax=lmax)
    #Fisher_Xi = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_Xi_filename']), 
    #                   kmin=RSDPower.KMIN, kmax=RSDPower.KMAX, lmax=lmax)
    #Fisher_tot = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_tot_filename']),
    #                   kmin=RSDPower.KMIN, kmax=RSDPower.KMAX, lmax=lmax)
    
    Fisher_P =  np.genfromtxt(params['fisher_bandpower_P_filename'])
    Fisher_Xi = np.genfromtxt(params['fisher_bandpower_Xi_filename'])
    #Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])
    
    Fisher_P_re = _reordering(Fisher_P, l=l)
    Fisher_Xi_re = _reordering(Fisher_Xi, l=l)
    #Fisher_tot_re = _reordering(Fisher_tot, l=l)

    if 'com' in params['probe']:
        Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])
        Fisher_tot_re = _reordering(Fisher_tot, l=l)
    # calculating SNRP


    
    if 'p' in params['probe']:

        FP = Fisher_P_re.copy()
        PP = datav_multipole_kcut_re.copy()

        SNRlist_P = []
        SNRP = np.dot( np.dot(PP, FP), PP.T )
        #print 'snrp first ', SNRP
        SNRlist_P.append(SNRP)
        for j in range(1, PP.shape[1]/l):
            PP = PP[:,:int(-1*l)]
            for i in range(0,l):
                FP = blockwise( FP )
            SNRP = np.dot( np.dot(PP, FP), PP.T )
            SNRlist_P.append(SNRP)
        SNRlist_P = np.array(SNRlist_P[::-1]).ravel()
        kklist3 = np.hstack([RSDPower.kcenter_y, RSDPower.kcenter_y, RSDPower.kcenter_y]).reshape(1, RSDPower.kcenter_y.size*3)
        masked_kbin = masking_paramsdatav(RSDPower, kklist3, kmin=kmin, kmax=kmax, lmax=0).ravel()
        #masked_kbin = masked_kbin_[:masked_kbin_.size/3].ravel()
                                
        DAT = np.column_stack((masked_kbin, SNRlist_P ))
        np.savetxt(params['savedir']+'snr_p.txt', DAT)
        print 'snr data saved to ', params['savedir']+'snr_p.txt'
    
    # Xi
    if 'xi' in params['probe']:

        F = Fisher_Xi_re.copy()
        P = datav_multipole_re.copy()

        SNRlist = []
        SNR = np.dot( np.dot(P, F), P.T )
        #print 'snrxi first ', SNR
        SNRlist.append(SNR)
        for j in range(1, P.shape[1]/l):
            P = P[:,:int(-1*l)]
            for i in range(0,l):
                F = blockwise( F )
            SNR = np.dot( np.dot(P, F), P.T )
            SNRlist.append(SNR)

        SNRlist = np.array(SNRlist[::-1]).ravel()
        DAT = np.column_stack((RSDPower.kcenter_y, SNRlist))
        np.savetxt(params['savedir']+'snr_xi.txt', DAT)
        print 'snr data saved to ', params['savedir']+'snr_xi.txt'
    
    # tot
    if 'com' in params['probe']:
        F = Fisher_tot_re.copy()
        P = datav_multipole_re.copy()

        SNRlist = []
        SNR = np.dot( np.dot(P, F), P.T )
        SNRlist.append(SNR)
        for j in range(1, P.shape[1]/l):
            P = P[:,:int(-1*l)]
            for i in range(0,l):
                F = blockwise( F )
            SNR = np.dot( np.dot(P, F), P.T )
            SNRlist.append(SNR)

        SNRlist = np.array(SNRlist[::-1]).ravel()
        DAT = np.column_stack((RSDPower.kcenter_y, SNRlist))
        np.savetxt(params['savedir']+'snr_tot.txt', DAT)
        print 'snr data saved to ', params['savedir']+'snr_tot.txt'
    
    #### end #######################################
    
    
    
def Fisher_params(params, RSDPower, parameter = [0,1,2,3], kmin=None, kmax=None, lmax=4):
    
    """
    parameter : parameter index that you want to include in Fisher matrix
    """
    # calling bandpoewr fisher
    Fisher_P = np.genfromtxt(params['fisher_bandpower_P_filename'])
    Fisher_Xi = np.genfromtxt(params['fisher_bandpower_Xi_filename'])
    #Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])

    
    # calling params datavector
    params_datav = np.genfromtxt(params['params_datavector_filename'])
                       
    # masking params datavector
   
    params_datav_mar = np.vstack(([ params_datav[p,:] for p in parameter] ))
    params_datav_mar_kcut = masking_paramsdatav(RSDPower, params_datav_mar, kmin=kmin, kmax=kmax, lmax=lmax)
    params_datav_mar = masking_paramsdatav(RSDPower, params_datav_mar, lmax=lmax, kmin = RSDPower.KMIN, kmax = RSDPower.KMAX)
    
    # projecting to params space
    F_params_P = np.dot( np.dot( params_datav_mar_kcut, Fisher_P), params_datav_mar_kcut.T)
    F_params_Xi = np.dot( np.dot( params_datav_mar, Fisher_Xi), params_datav_mar.T )
    #F_params_tot = np.dot( np.dot( params_datav_mar, Fisher_tot), params_datav_mar.T )
    

    print 'Fisher_params diagonal check',
    print np.sum(F_params_P.diagonal() < 0), np.sum(F_params_Xi.diagonal() < 0), # np.sum(F_params_tot.diagonal() < 0)
    

    from numpy.linalg import pinv as inv
    sigP = np.sqrt(inv( F_params_P ).diagonal())
    sigXi = np.sqrt(inv( F_params_Xi ).diagonal())
    #sigtot = np.sqrt(inv( F_params_tot ).diagonal())
    sigdiff = np.sqrt(inv( F_params_P + F_params_Xi ).diagonal())


    if 'com' in params['probe']:

        Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])
        F_params_tot = np.dot( np.dot( params_datav_mar, Fisher_tot), params_datav_mar.T )
        print np.sum(F_params_tot.diagonal() < 0)
        sigtot = np.sqrt(inv( F_params_tot ).diagonal())




    parameter_name = ['b', 'f', 's', 'nn']

    print '\n\nname  :', 
    for i in range(len(parameter)): print parameter_name[i],

    print '\nP  :',
    for i in range(len(parameter)): print sigP[i],

    print '\nXi :',
    for i in range(len(parameter)): print sigXi[i],

    if 'com' in params['probe']:
        print '\ncom:',
        for i in range(len(parameter)): print sigtot[i],

    print '\ndif:',
    for i in range(len(parameter)): print sigdiff[i],

    print '\n'



    params['fisher_params_p'] = F_params_P
    params['fisher_params_Xi'] = F_params_Xi

    params['cov_params_p'] = inv(F_params_P)
    params['cov_params_Xi'] = inv(F_params_Xi)
    params['cov_params_diff'] =  inv(F_params_P+F_params_Xi)

    params['sigma_params_p'] = sigP 
    params['sigma_params_Xi'] = sigXi
    params['sigma_params_diff'] = sigdiff

    if 'com' in params['probe']:
        params['fisher_params_tot'] = F_params_tot
        params['cov_params_tot'] = inv(F_params_tot)
        params['sigma_params_tot'] = sigtot 

    #print '\nname     p    xi    com    diff'
    #for i in range(len(parameter)):
    #    print parameter_name[i], ':', sigP[i], sigXi[i], sigtot[i], sigdiff[i]

    

    #ind = np.arange(0,len(parameter)**2)
    #DAT = np.column_stack((ind, F_params_P.ravel(), F_params_Xi.ravel(), F_params_tot.ravel() ))
    #np.savetxt(params['savedir']+'fisher_params.txt', DAT)
    #print '\nfile save to ', params['savedir']+'fisher_params.txt'
                       
                       
    
def Reid_error(params, RSDPower, parameter = None, lmax=None):
    
    from test_SNR import reordering, reorderingVector
    
    if lmax == 0 : l = 1
    elif lmax == 2 : l = 2
    elif lmax == 4 : l = 3
    else : 
        print 'invalid lmax'
        return None

    ## calling stored cov and datavector
    #covPP = np.genfromtxt(params['covPP_filename'])                   
    covXi = np.genfromtxt(params['covXi_filename'])

    #covPP_re, _ = reordering( RSDPower, covPP )
    covXi_re, _ = reordering( RSDPower, covXi, lmax=lmax, xi=True )

    datav_Xi = np.genfromtxt(params['derivative_Xi_filename'])
    datav_Xi_re = reorderingVector(datav_Xi, lmax=lmax)
    
    # params_datav
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    params_datav_mar = np.vstack(( [params_datav[p,:] for p in parameter] ))
    params_datav_mar_re = reorderingVector(params_datav_mar, lmax=lmax)
    
    if 2 in parameter : 
        print 'parameter sv is marginalized'
    elif 3 in parameter : 
        print 'parameter sv, 1/n are marginalized'

        
    # from P #####################################

    if 'p' in params['probe']:
        errPb, errPf = [], []

        rlistP = []
        
        Fisher_P = np.genfromtxt(params['fisher_bandpower_P_filename'])
        re_F,_ = reordering( RSDPower, Fisher_P, lmax=lmax )
        #re_F = inv(covPP_re)
        re_dP = params_datav_mar_re.copy()
        
        FPparams = np.dot( np.dot(re_dP, re_F), re_dP.T )      
        CPparams = inv(FPparams)
        sigma_Pb, sigma_Pf = CPparams[0,0], CPparams[1,1]
        errPb.append(np.sqrt(sigma_Pb)/RSDPower.b)
        errPf.append(np.sqrt(sigma_Pf)/RSDPower.f)

        rlistP.append( 1.15 * np.pi/RSDPower.kcenter_y[::-1][0])


        for j in range(1, RSDPower.kcenter_y.size ):
            
            re_F = re_F[:-1*l, :-1*l]
            rlistP.append( 1.15 * np.pi/RSDPower.kcenter_y[::-1][j])

            # bf
            re_dP = re_dP[:,:-1*l]
            FPparams = np.dot( np.dot(re_dP, re_F), re_dP.T )
            CPparams = inv(FPparams)
            sigma_Pb, sigma_Pf = CPparams[0,0], CPparams[1,1]
            errPb.append(np.sqrt(sigma_Pb)/RSDPower.b)
            errPf.append(np.sqrt(sigma_Pf)/RSDPower.f)

        DAT = np.column_stack((rlistP, errPb, errPf))
        np.savetxt(params['savedir']+'reid_p.txt', DAT, header = ' rcenter, errb, errf ')    
        print 'reid data saved to ', params['savedir']+'reid_p.txt'    
        
        
    ### from Xi ########################################
    
    if 'xi' in params['probe']:
        # err list
        errb, errf= [], []
        rlist = [] 

        #=== First term (all scale of r) =====================
        re_C = covXi_re.copy()
        DXIP = datav_Xi_re.copy()
        dP = params_datav_mar.copy()
        
        F = np.dot( np.dot(DXIP, inv(re_C)), DXIP.T ) 
        
        Fparams = np.dot( np.dot(dP, F), dP.T )
        Cparams = inv(Fparams)
        sigma_b, sigma_f = Cparams[0,0], Cparams[1,1]
        errb.append(np.sqrt(sigma_b)/RSDPower.b)
        errf.append(np.sqrt(sigma_f)/RSDPower.f)
        reverser = RSDPower.rcenter[::-1]
        rlist.append(reverser[0])

        #=== now from 2nd term =====================
        for j in range(1, RSDPower.rcenter.size):

            DXIP = DXIP[:, :-1*l ]
            re_C = re_C[:-1*l, :-1*l ]
            F = np.dot( np.dot(DXIP, inv(re_C)), DXIP.T )

            # shot noise determined
            Fparams = np.dot( np.dot(dP, F), dP.T )
            Cparams = inv(Fparams)
            sigma_b, sigma_f = Cparams[0,0], Cparams[1,1]
            errb.append(np.sqrt(sigma_b)/RSDPower.b)
            errf.append(np.sqrt(sigma_f)/RSDPower.f)
            rlist.append(reverser[j])
            

        DAT = np.column_stack((rlist, errb, errf))
        np.savetxt(params['savedir']+'reid_xi.txt', DAT, header = ' rcenter, errb, errf ')
        print 'reid data saved to', params['savedir']+'reid_xi.txt' 
    
    #### end ###################################
       

def save_data_to_fits(params):

    
    ResultDic = {}
    
    #ResultDic['kcenter'] = params['kcenter']
    #ResultDic['kbin'] = params['kbin']
    #ResultDic['rcenter'] = params['rcenter']
    #ResultDic['rbin'] = params['rbin']
    """
    ResultDic['rbin'] =
    ResultDic['rbin'] =
    ResultDic['rbin'] =
    ResultDic['rbin'] =
    """


    if params['direct_projection']:
        ResultDic['fisher_params_p_direct'] = params['fisher_params_p_direct']
        ResultDic['fisher_params_Xi_direct'] = params['fisher_params_Xi_direct']
        #ResultDic['fisher_params_tot'] = params['fisher_params_tot']

        ResultDic['cov_params_p_direct'] = params['cov_params_p_direct']
        ResultDic['cov_params_Xi_direct'] = params['cov_params_Xi_direct'] 
        #ResultDic['cov_params_tot'] = params['cov_params_tot'] 
        ResultDic['cov_params_diff_direct'] = params['cov_params_diff_direct'] 

        ResultDic['sigma_params_p_direct'] = params['sigma_params_p_direct'] 
        ResultDic['sigma_params_Xi_direct'] = params['sigma_params_Xi_direct'] 
        #ResultDic['sigma_params_tot'] = params['sigma_params_tot'] 
        ResultDic['sigma_params_diff_direct'] = params['sigma_params_diff_direct'] 
        

        if 'com' in params['probe']:
            ResultDic['fisher_params_tot_direct'] = params['fisher_params_tot_direct']
            ResultDic['cov_params_tot_direct'] = params['cov_params_tot_direct'] 
            ResultDic['sigma_params_tot_direct'] = params['sigma_params_tot_direct'] 


    else:
        ResultDic['fisher_params_p'] = params['fisher_params_p']
        ResultDic['fisher_params_Xi'] = params['fisher_params_Xi']
        #ResultDic['fisher_params_tot'] = params['fisher_params_tot']

        ResultDic['cov_params_p'] = params['cov_params_p']
        ResultDic['cov_params_Xi'] = params['cov_params_Xi'] 
        #ResultDic['cov_params_tot'] = params['cov_params_tot'] 
        ResultDic['cov_params_diff'] = params['cov_params_diff'] 

        ResultDic['sigma_params_p'] = params['sigma_params_p'] 
        ResultDic['sigma_params_Xi'] = params['sigma_params_Xi'] 
        #ResultDic['sigma_params_tot'] = params['sigma_params_tot'] 
        ResultDic['sigma_params_diff'] = params['sigma_params_diff'] 
        

        if 'com' in params['probe']:
            ResultDic['fisher_params_tot'] = params['fisher_params_tot']
            ResultDic['cov_params_tot'] = params['cov_params_tot'] 
            ResultDic['sigma_params_tot'] = params['sigma_params_tot'] 

    fitsname = params['savedir']+'output_'+params['name']+'.fits'
    fitsio.write(fitsname, ResultDic, clobber=True)
    print 'save to ', fitsname

    return 0

                        
########## main #######
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='call run_error_analysis outside the pipeline')
    parser.add_argument("parameter_file", help="YAML configuration file")
    args = parser.parse_args()
    try:
        param_file = args.parameter_file   
    except SystemExit:
        sys.exit(1)


    params = yaml.load(open(param_file))
    os.system('cp '+param_file+' '+params['savedir']+'/params.yaml')

    print '-----------------------------------'
    print ' Run Error Analaysis'
    print '-----------------------------------'


    params['savedir'] = params['savedir']+'/'
    save_dir = params['savedir']

    if os.path.exists(save_dir) : 
        print " savedir exists... "
    else : 
        print " making savedir... "
        os.makedirs(save_dir)
    print ' savedir : ', save_dir
    run_error_analysis(params)
    save_data_to_fits(params)

    #stream = file(params['savedir'] + '/' + params['name']+'.yaml', 'w')
    #yaml.dump( params, stream )
    print '\n------------------ end ---------------------'

