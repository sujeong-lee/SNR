import time, datetime
import numpy as np
from numpy import zeros, sqrt, pi, vectorize
from numpy.linalg import pinv, inv
from multiprocessing import Process, Queue
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from error_analysis_class import *
from noshellavg import *
from matplotlib.backends.backend_pdf import PdfPages


import sys
import argparse
import yaml
#from run_cosmolike_mpp import *


    
def run_error_analysis(params):
 
    kmin, kmax, kN = params['k']
    rmin, rmax, rN = params['r']
    logscale = params['logscale']
    KMIN, KMAX = 1e-10, 2.0
    lmax = params['lmax']
    parameter_ind = params['parameter_ind']   
    
    RSDPower = NoShell_covariance(KMIN, KMAX, rmin, rmax, 2**10 + 1, rN, 1, kN, logscale = logscale)
    Covariance_matrix(params, RSDPower)
    
    
    if 'multipole_p_filename' not in params:
        P_multipole(params, RSDPower)
    else : print '\nUse Precalculated multipole_p ', params['multipole_p_filename']
        
    if 'derivative_P_filename' not in params:
        derivative_P_datavector(params, RSDPower)
    else : print 'Use Precalculated derivative_P ', params['derivative_P_filename']     
        
        
    if 'derivative_Xi_filename' not in params:
        derivative_Xi_datavector(params, RSDPower)
    else : print 'Use Precalculated derivative_Xi ', params['derivative_Xi_filename']    
      
    if 'params_datavector_filename' not in params:
        params_datavector(params, RSDPower)
    else : print '\nUse Precalculated params_datavector ', params['params_datavector_filename']     
        
    
    BandpowerFisher(params, RSDPower, kmin = kmin, kmax = kmax, lmax = lmax) 
    Fisher_params(params, RSDPower, parameter = parameter_ind, kmin=kmin, kmax=kmax, lmax=lmax)
    
    direct_projection = 0
    if 'direct_projection' in params:
        direct_projection = params['direct_projection']
    if direct_projection :
        if 'params_xi_datavector_filename' not in params:
            params_xi_datavector(params, RSDPower)
        else : 
            print '\nUse Precalculated params_si_datavector ', params['params_xi_datavector_filename'] 
        DirectProjection_to_params(params, RSDPower, parameter =parameter_ind, kmin=kmin, kmax=kmax)
    
    SNR = params['SNR']
    if SNR : 
        print 'calculating SNR...'
        CumulativeSNR(params, RSDPower, kmin=kmin, kmax=kmax)

    Reid = params['Reid']  
    if Reid : 
        print 'calclating Reid result...'
        Reid_error(params, RSDPower, parameter = parameter_ind)
    
    
    print '------------------ end ---------------------'

    
def Covariance_matrix(params, RSDPower):    
    
    name = params['name']

    file = 'matterpower_z_0.55.dat'  # from camb (z=0.55)
    RSDPower.MatterPower(file = file)
    RSDPower.multipole_P_band_all()
    # P covariance matrix ( nine submatrices C_ll' )
    
    
    if 'covPP_filename' not in params:
        RSDPower.RSDband_covariance_PP_all()
        C_matrix3PP = np.vstack((
                np.hstack([RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04 ]),\
                np.hstack([RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24 ]),\
                np.hstack([RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44 ])
                ))
        f = 'data_txt/cov/'+params['name']+'_PP.cov'
        np.savetxt(f, C_matrix3PP)
        params['covPP_filename'] = f
    else : print 'Use Precalculated CovPP ', params['covPP_filename']
        
    if 'covXi_filename' not in params:
        RSDPower.covariance_Xi_all()
        C_matrix3Xi = np.vstack((
                np.hstack([RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04 ]),\
                np.hstack([RSDPower.covariance02.T, RSDPower.covariance22, RSDPower.covariance24 ]),\
                np.hstack([RSDPower.covariance04.T, RSDPower.covariance24.T, RSDPower.covariance44 ])
                ))
        f2 = 'data_txt/cov/'+params['name']+'_Xi.cov'
        np.savetxt(f2, C_matrix3Xi) 
        params['covXi_filename'] = f2
    else : print 'Use Precalculated CovXi ', params['covXi_filename']
        
    if 'covPXi_filename' not in params:
        RSDPower.covariance_PXi_All()
        C_matrix3PXi = np.vstack((
                np.hstack([RSDPower.covariance_PXi00, RSDPower.covariance_PXi02, RSDPower.covariance_PXi04 ]),\
                np.hstack([RSDPower.covariance_PXi20, RSDPower.covariance_PXi22, RSDPower.covariance_PXi24 ]),\
                np.hstack([RSDPower.covariance_PXi40, RSDPower.covariance_PXi42, RSDPower.covariance_PXi44 ])
                ))
        
        f3 = 'data_txt/cov/'+params['name']+'_PXi.cov'
        np.savetxt(f3, C_matrix3PXi)
        params['covPXi_filename'] = f3
    else : print 'Use Precalculated CovPXi ', params['covPXi_filename']

    ##### end #### -----------------------------------------------------------------------
    
def P_multipole(params, RSDPower):

    # power spectrum multipoles l = 0,2,4
    
    RSDPower.multipole_P_band_all()
    multipole_datav = np.hstack([RSDPower.multipole_bandpower0,RSDPower.multipole_bandpower2\
                             ,RSDPower.multipole_bandpower4])
    np.savetxt('data_txt/datav/'+params['name']+'_multipole_p.datavector',multipole_datav )
    params['multipole_p_filename'] = 'data_txt/datav/'+params['name']+'_multipole_p.datavector'
           
def derivative_P_datavector(params, RSDPower):    

    derivative_P0 = np.identity(RSDPower.kcenter_y.size)# [:,kcut_min:kcut_max+1]
    Pzeros = np.zeros((derivative_P0.shape))

    derivative_P = np.concatenate((np.concatenate((derivative_P0, Pzeros, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, derivative_P0, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, Pzeros, derivative_P0),axis=1 )), axis=0)
    f = 'data_txt/datav/'+params['name']+'_P.datavector'
    np.savetxt(f,derivative_P)
    params['derivative_P_filename'] = f
    
        
def derivative_Xi_datavector(params, RSDPower):        

    RSDPower.derivative_Xi_band_all()
    Xizeros = np.zeros((RSDPower.dxip0.shape))
    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0,Xizeros,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,RSDPower.dxip2,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,Xizeros,RSDPower.dxip4), axis=1)),axis=0 )

    f2 = 'data_txt/datav/'+params['name']+'_Xi.datavector'
    np.savetxt(f2,derivative_correl_avg)
    params['derivative_Xi_filename'] = f2

    ## end #####################################################################
    
    
def params_datavector(params, RSDPower):
    
    #if 'params_datavector_filename' not in params:
        
    # derivative dXidb, s, f, n
    #RSDPower.derivative_bfs_all()
    RSDPower.derivative_P_bfs_all()

    # add shot noise params
    dPN0 = np.ones(RSDPower.kcenter_y.size)
    dPN1 = np.zeros(RSDPower.kcenter_y.size)
    dPN2 = dPN1.copy()

    matrices2P = np.vstack((
            np.hstack([RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4]),\
            np.hstack([RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4]),\
            np.hstack([RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]),\
            np.hstack([dPN0, dPN1, dPN2]) ))


    f = 'data_txt/datav/'+params['name']+'_params.datavector'
    np.savetxt(f, matrices2P)
    params['params_datavector_filename'] = f
    #else : print 'Use Precalculated params_datavector ', params['params_datavector_filename']

    ### end ####################################################################        
    

def params_xi_datavector(params, RSDPower):
    
    #if 'params_datavector_filename' not in params:
        
    # derivative dXidb, s, f, n
    RSDPower.derivative_bfs_all()
    #RSDPower.derivative_P_bfs_all()

    
    kmax = RSDPower.KMAX
    kmin = RSDPower.KMIN
    r = RSDPower.rcenter
    
    dxin0 = (-kmax*r*np.cos(kmax*r) + kmin*r*np.cos(kmin*r) + np.sin(kmax*r) - 
 np.sin(kmin*r))/(2*np.pi**2* r**3)
    
    dxin2 = np.zeros(RSDPower.rcenter.size)
    dxin4 = dxin2.copy()
    
    
    matrices2P = np.vstack((
            np.hstack([RSDPower.dxib0, RSDPower.dxib2, RSDPower.dxib4]),\
            np.hstack([RSDPower.dxif0, RSDPower.dxif2, RSDPower.dxif4]),\
            np.hstack([RSDPower.dxis0, RSDPower.dxis2, RSDPower.dxis4]),\
            np.hstack([dxin0, dxin2, dxin4]) ))


    f = 'data_txt/datav/'+params['name']+'_params_xi.datavector'
    np.savetxt(f, matrices2P)
    params['params_xi_datavector_filename'] = f
    #else : print 'Use Precalculated params_datavector ', params['params_datavector_filename']

    ### end ####################################################################        

    
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
    

    
def BandpowerFisher(params, RSDPower, kmin = 0, kmax = 2, lmax=4):
    
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    covPP_masked = masking(RSDPower, covPP, kmin = kmin, kmax = kmax, lmax=lmax)
    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax)
    
    C_tot = np.concatenate((np.concatenate((covPP_masked, covPXi), axis=1),
                            np.concatenate((covPXi.T, covXi), axis=1)), axis = 0)

    datav_P = masking_datav(RSDPower, np.genfromtxt(params['derivative_P_filename']), kmin = kmin, kmax = kmax, lmax=lmax)
    datav_Xi = masking_datav(RSDPower, np.genfromtxt(params['derivative_Xi_filename']), xi=True, lmax=lmax)
    datav = np.concatenate((datav_P,datav_Xi), axis=1)
    
    # inverting matrices
    #from test_SNR import blockwiseInversion

    
    if 'fisher_bandpower_P_filename' not in params:               
        #FisherP = pinv(covPP)
        
        cut = RSDPower.kcenter_y.size
        covPPlist = [covPP[:cut, :cut], covPP[:cut, cut:2*cut], covPP[:cut, 2*cut:],
                    covPP[:cut, cut:2*cut], covPP[cut:2*cut, cut:2*cut], covPP[cut:2*cut, 2*cut:], 
                    covPP[:cut, 2*cut:], covPP[cut:2*cut, 2*cut:], covPP[2*cut:, 2*cut:]]
        FisherP = masking(RSDPower, DiagonalBlockwiseInversion3x3(*tuple(covPPlist)), kmin=kmin, kmax=kmax, lmax=lmax)
                           
        #FisherBand_P = FisherProjection_Fishergiven(datav_P, FisherP)
        #FisherBand_P = np.dot(np.dot(datav_P, FisherP), datav_P.T)
        f = 'data_txt/cov/'+params['name']+'_bandpower_PP.fisher'
        np.savetxt(f, FisherP)
        params['fisher_bandpower_P_filename']= f
        print '\nFisherP saved ', f
        
    else : print '\nUse Precalculated FisherP ', params['fisher_bandpower_P_filename']
        
    if 'fisher_bandpower_Xi_filename' not in params:
        FisherXi = masking(RSDPower, pinv(covXi, rcond=1e-30), xi=True, lmax=lmax)
        #print '\nFisherXi', np.sum(covXi.diagonal() <= 0.0)
        FisherBand_Xi = FisherProjection_Fishergiven(datav_Xi, FisherXi)
        #FisherBand_Xi = np.dot(np.dot(datav_Xi, FisherXi), datav_Xi.T)
        f2 = 'data_txt/cov/'+params['name']+'_bandpower_Xi.fisher'
        np.savetxt(f2, FisherBand_Xi)
        params['fisher_bandpower_Xi_filename']= f2   
        print 'FisherXi saved ', f2
    else : print 'Use Precalculated FisherXi ', params['fisher_bandpower_Xi_filename']
        
    if 'fisher_bandpower_tot_filename' not in params:    
    
        print 'calculating Fisher tot'
        # blockwise inversion
        #from numpy.linalg import pinv
        
        
        FisherP = np.genfromtxt(params['fisher_bandpower_P_filename'])
        b = covPXi
        c = covPXi.T 
        d = covXi
        ia = FisherP
        
        import time
        t1 = time.time()
        Fd = pinv( d - np.dot( np.dot( c, ia ), b), rcond=1e-30)
        #Fc = - np.dot( np.dot( Fd, c), ia)
        Fb = - np.dot( np.dot( ia, b ), Fd )
        #Fc = Fb.T
        Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)
        
        Fa = masking(RSDPower, Fa, kmin=kmin, kmax=kmax, lmax=lmax)
        Fb = masking(RSDPower, Fb, kmin=kmin, kmax=kmax, lmax=lmax)
        Fc = Fb.T
        Fd = masking(RSDPower, Fd, xi=True, lmax=lmax)
        print time.time()-t1
        Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
        
        
        #Fisher3_tot = pinv(C_tot, rcond = 1e-30)
        """
        rcondnum = np.arange(30, 13, -1)
        for rc in rcondnum:
            Fisher3_tot = pinv(C_tot, rcond = 10**(-1*rc))
            neg = np.sum(Fisher3_tot.diagonal() <=0.0)
            print rc, neg
            if neg == 0 : break
        if rc == 13 : raise ValueError("Inversion failed : rcond exceeds 1e-15")   
        
        """
        
        #FisherBand_tot = np.dot( np.dot( datav, Fisher3_tot), datav.T)
        FisherBand_tot = FisherProjection_Fishergiven(datav, Fisher3_tot)
        print np.sum(FisherBand_tot.diagonal() <=0.0)
        f3 = 'data_txt/cov/'+params['name']+'_bandpower_tot.fisher'
        np.savetxt(f3, FisherBand_tot)
        params['fisher_bandpower_tot_filename']= f3
        print 'Fishertot saved ', f3
    else : print 'Use Precalculated Fisher_tot ', params['fisher_bandpower_tot_filename']

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

def DirectProjection_to_params(params, RSDPower, parameter =[0,1,2,3], kmin = 0, kmax = 2, diffs = True):
    
    ## calling stored cov and datavector
    covPP = np.genfromtxt(params['covPP_filename'])
    covXi = masking(RSDPower, np.genfromtxt(params['covXi_filename']), xi=True, lmax=lmax)
    covPXi = masking(RSDPower, np.genfromtxt(params['covPXi_filename']), kmin = kmin, kmax = kmax, lmax=lmax)
    
    #covPP_masked = masking(RSDPower, covPP, kmin = kmin, kmax = kmax)
    #C_tot = np.concatenate((np.concatenate((covPP_masked, covPXi), axis=1),\
    #                                np.concatenate((covPXi.T, covXi), axis=1)), axis = 0)
    
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    params_datav_mar = np.vstack(([ params_datav[p,:] for p in parameter] ))
    params_datav_mar_kcut = masking_datav(RSDPower, params_datav_mar, kmin=kmin, kmax=kmax, lmax=lmax)
      
    params_xi_datav = np.genfromtxt(params['params_xi_datavector_filename'])
    params_xi_datav_mar = np.vstack(([ params_xi_datav[p,:] for p in parameter] ))
    
    if diffs : 
        dpss = np.zeros(params_datav.shape[1])
        params_datav_mar_kcut = np.insert(params_datav_mar_kcut, 3, dpss, axis=0 )
        
        dxiss = np.zeros(params_xi_datav.shape[1])
        params_xi_datav_mar = np.insert(params_xi_datav_mar, 2, dxiss, axis=0 )
    
    datav = np.concatenate((params_datav_mar_kcut,params_xi_datav_mar), axis=1)
    
    
    # inverting matrices
    from test_SNR import blockwiseInversion
    
    if 'fisher_bandpower_P_filename' not in params: FisherP = pinv(covPP)
    else : FisherP = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_P_filename']), kmin=kmin, kmax=kmax, lmax=lmax)
                             
    F_params_P = np.dot(np.dot(params_datav_mar_kcut,FisherP), params_datav_mar_kcut.T)
    
    if 'fisherXi_filename' not in params: FisherXi = pinv(covXi)
    else : FisherXi = np.genfromtxt(params['fisherXi_filename'])
    F_params_Xi = np.dot(np.dot(params_xi_datav_mar, FisherXi), params_xi_datav_mar.T)
        
    if 'fishertot_filename' not in params:    

        print 'calculating Fisher tot'
        
        b = covPXi
        c = covPXi.T #matrix[cutInd+1:, 0:cutInd+1]
        d = covXi
        ia = masking(RSDPower, FisherP, kmin=kmin, kmax=kmax)

        Fd = pinv( d - np.dot( np.dot( c, ia ), b) )
        Fc = - np.dot( np.dot( Fd, c), ia)
        Fb = - np.dot( np.dot( ia, b ), Fd )
        Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)

        Fisher3_tot = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))

        """
        #Fisher3_tot = pinv(C_tot, rcond = 1e-30)

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
    if diffs == True : ind = np.arange(0,(len(parameter+1))**2)
    DAT = np.column_stack((ind, F_params_P.ravel(), F_params_Xi.ravel(), F_params_tot.ravel() ))
    np.savetxt('data_txt/'+params['name']+'_fisher_params_direct.txt', DAT)
    print 'save to', 'data_txt/'+params['name']+'_fisher_params_direct.txt'
    ##### end #########################################


    
    
    
def _reorderingVector( vector ):
    
    if len(vector) > 1: vector = np.hstack(vector)
        
    try : 
        nx, ny = vector.shape
        
        ReorderedP = np.zeros((vector.shape))
        ind = np.arange(0, ny, 3)

        Ny = ny/3
        for i in range(3):
            ReorderedP[i,:][ind] = vector[i,:Ny]
            
            ReorderedP[i,:][ind+1] = vector[i,Ny:Ny*2]
            ReorderedP[i,:][ind+2] = vector[i,Ny*2:Ny*3]
            
         
    except : 
 
        Nx = vector.size/3
        ReorderedP = np.zeros((vector.shape))
        ind = np.arange(0,vector.size, 3)

        Nx = vector.size/3
        ReorderedP[ind] = vector[:Nx]
        ReorderedP[ind+1] = vector[Nx:Nx*2]
        ReorderedP[ind+2] = vector[Nx*2:Nx*3]
    
    return ReorderedP
    

   
    
def CumulativeSNR(params, RSDPower, kmin=0, kmax=2, lmax=4):
    
    
    from test_SNR import reorderingVector, reordering, blockwise
    
    multipole_p = np.genfromtxt(params['multipole_p_filename'])
    #datav_multipole = masking_datav(RSDPower, multipole_p.reshape(1, multipole_p.size)
    #                                ,kmin=kmin, kmax=kmax, lmax=lmax)    
    datav_multipole_re = reorderingVector(multipole_p.reshape(1, multipole_p.size))

    ## loading fisher matrix
    Fisher_P = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_P_filename']), 
                       kmin=kmin, kmax=kmax, lmax=lmax)
    Fisher_Xi = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_Xi_filename']), 
                       kmin=RSDPower.KMIN, kmax=RSDPower.KMAX, lmax=lmax)
    Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])
    
    Fisher_P_re, _ = reordering( RSDPower, Fisher_P)
    Fisher_Xi_re, _ = reordering( RSDPower, Fisher_Xi)
    Fisher_tot_re, _ = reordering( RSDPower, Fisher_tot)
       
    # calculating SNRP

    FP = Fisher_P_re.copy()
    PP = datav_multipole_re.copy()
    
    SNRlist_P = []
    SNRP = np.dot( np.dot(PP, FP), PP.T )
    SNRlist_P.append(SNRP)
    for j in range(1, PP.size/3):
        PP = PP[:,:-3]
        for i in range(0,3):
            FP = blockwise( FP )
        SNRP = np.dot( np.dot(PP, FP), PP.T )
        SNRlist_P.append(SNRP)

    SNRlist_P = np.array(SNRlist_P[::-1]).ravel()
    kklist = RSDPower.kcenter_y
    DAT = np.column_stack((RSDPower.kcenter_y, SNRlist_P))
    np.savetxt('data_txt/snr/'+params['name']+'_snr_p', DAT)
    print 'snr data saved to ', 'data_txt/snr/'+params['name']+'_snr_p'
    
    # Xi

    F = Fisher_Xi_re.copy()
    P = datav_multipole_re.copy()
    
    SNRlist = []
    SNR = np.dot( np.dot(P, F), P.T )
    SNRlist.append(SNR)
    for j in range(1, P.size/3):
        P = P[:,:-3]
        for i in range(0,3):
            F = blockwise( F )
        SNR = np.dot( np.dot(P, F), P.T )
        SNRlist.append(SNR)

    SNRlist = np.array(SNRlist[::-1]).ravel()
    DAT = np.column_stack((RSDPower.kcenter_y, SNRlist))
    np.savetxt('data_txt/snr/'+params['name']+'_snr_xi', DAT)
    print 'snr data saved to ', 'data_txt/snr/'+params['name']+'_snr_xi'
    
    # tot

    F = Fisher_tot_re.copy()
    P = datav_multipole_re.copy()

    SNRlist = []
    SNR = np.dot( np.dot(P, F), P.T )
    SNRlist.append(SNR)
    for j in range(1, P.size/3):
        P = P[:,:-3]
        for i in range(0,3):
            F = blockwise( F )
        SNR = np.dot( np.dot(P, F), P.T )
        SNRlist.append(SNR)

    SNRlist = np.array(SNRlist[::-1]).ravel()
    DAT = np.column_stack((RSDPower.kcenter_y, SNRlist))
    np.savetxt('data_txt/snr/'+params['name']+'_snr_tot', DAT)
    print 'snr data saved to ', 'data_txt/snr/'+params['name']+'_snr_tot'
    
    #### end #######################################
    
    
    
def Fisher_params(params, RSDPower, parameter = [0,1,2,3], kmin=0, kmax=2, lmax=4):
    
    """
    parameter : parameter index that you want to include in Fisher matrix
    """
    # calling bandpoewr fisher
    Fisher_P = masking(RSDPower, np.genfromtxt(params['fisher_bandpower_P_filename']), kmin=kmin, kmax=kmax)
    Fisher_Xi = np.genfromtxt(params['fisher_bandpower_Xi_filename'])
    Fisher_tot = np.genfromtxt(params['fisher_bandpower_tot_filename'])
    
    # calling params datavector
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    
                       
    # masking params datavector
   
    params_datav_mar = np.vstack(([ params_datav[p,:] for p in parameter] ))
    params_datav_mar_kcut = masking_datav(RSDPower, params_datav_mar, kmin=kmin, kmax=kmax, lmax=lmax)
    
    # projecting to params space
    F_params_P = np.dot( np.dot( params_datav_mar_kcut, Fisher_P), params_datav_mar_kcut.T)
    F_params_Xi = np.dot( np.dot( params_datav_mar, Fisher_Xi), params_datav_mar.T )
    F_params_tot = np.dot( np.dot( params_datav_mar, Fisher_tot), params_datav_mar.T)
          
    ind = np.arange(0,len(parameter)**2)
    DAT = np.column_stack((ind, F_params_P.ravel(), F_params_Xi.ravel(), F_params_tot.ravel() ))
    np.savetxt('data_txt/'+params['name']+'_fisher_params.txt', DAT)
    print 'file save to ', 'data_txt/'+params['name']+'_fisher_params.txt'
                       
                       
    
def Reid_error(params, RSDPower, parameter = [0,1,2,3]):
    
    from test_SNR import reordering, reorderingVector
    
    ## calling stored cov and datavector
    #covPP = np.genfromtxt(params['covPP_filename'])                   
    covXi = np.genfromtxt(params['covXi_filename'])

    #covPP_re, _ = reordering( RSDPower, covPP )
    covXi_re, _ = reordering( RSDPower, covXi )

    datav_Xi = np.genfromtxt(params['derivative_Xi_filename'])
    datav_Xi_re = reorderingVector(datav_Xi)
    
    # params_datav
    params_datav = np.genfromtxt(params['params_datavector_filename'])
    params_datav_mar = np.vstack(( [params_datav[p,:] for p in parameter] ))
    params_datav_mar_re = reorderingVector(params_datav_mar)
    
    if 2 in parameter : 
        print 'parameter sv is marginalized'
    elif 3 in parameter : 
        print 'parameter sv, 1/n are marginalized'

        
    # from P #####################################
    errPb, errPf = [], []

    rlistP = []
    
    Fisher_P = np.genfromtxt(params['fisher_bandpower_P_filename'])
    re_F,_ = reordering( RSDPower, Fisher_P )
    #re_F = inv(covPP_re)
    re_dP = params_datav_mar_re.copy()
    
    FPparams = np.dot( np.dot(re_dP, re_F), re_dP.T )      
    CPparams = inv(FPparams)
    sigma_Pb, sigma_Pf = CPparams[0,0], CPparams[1,1]
    errPb.append(np.sqrt(sigma_Pb)/RSDPower.b)
    errPf.append(np.sqrt(sigma_Pf)/RSDPower.f)

    rlistP.append( 1.15 * np.pi/RSDPower.kcenter_y[::-1][0])


    for j in range(1, RSDPower.kcenter_y.size ):
        
        re_F = re_F[:-3, :-3]
        rlistP.append( 1.15 * np.pi/RSDPower.kcenter_y[::-1][j])

        # bf
        re_dP = re_dP[:,:-3]
        FPparams = np.dot( np.dot(re_dP, re_F), re_dP.T )
        CPparams = inv(FPparams)
        sigma_Pb, sigma_Pf = CPparams[0,0], CPparams[1,1]
        errPb.append(np.sqrt(sigma_Pb)/RSDPower.b)
        errPf.append(np.sqrt(sigma_Pf)/RSDPower.f)

    DAT = np.column_stack((rlistP, errPb, errPf))
    np.savetxt('data_txt/reid/'+params['name']+'_reid_p.txt', DAT)    
    print 'reid data saved to ', 'data_txt/reid/'+params['name']+'_reid_p.txt'    
        
        
    ### from Xi ########################################
    
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

        DXIP = DXIP[:, :-3 ]
        re_C = re_C[:-3, :-3 ]
        F = np.dot( np.dot(DXIP, inv(re_C)), DXIP.T )

        # shot noise determined
        Fparams = np.dot( np.dot(dP, F), dP.T )
        Cparams = inv(Fparams)
        sigma_b, sigma_f = Cparams[0,0], Cparams[1,1]
        errb.append(np.sqrt(sigma_b)/RSDPower.b)
        errf.append(np.sqrt(sigma_f)/RSDPower.f)
        rlist.append(reverser[j])
        

    DAT = np.column_stack((rlist, errb, errf))
    np.savetxt('data_txt/reid/'+params['name']+'_reid_xi.txt', DAT)
    print 'reid data saved to ', 'data_txt/reid/'+params['name']+'_reid_xi.txt' 
    
    #### end ###################################
       


    


                        
########## main #######
if __name__=='__main__':

    import warnings

    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    parser = argparse.ArgumentParser(description='call run_error_analysis outside the pipeline')
    parser.add_argument("parameter_file", help="YAML configuration file")
    args = parser.parse_args()
    try:
        param_file = args.parameter_file   
    except SystemExit:
        sys.exit(1)

    params = yaml.load(open(param_file))
    run_error_analysis(params)

