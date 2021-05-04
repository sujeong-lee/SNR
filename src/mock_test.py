import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import scipy
from noshellavg_v2 import *
#def log_interp1d(xx, yy, kind='linear'):
#    logx = np.log10(xx)
#    logy = np.log10(yy)
#    lin_interp = interp1d(logx, logy, kind=kind)
#    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
#    return log_interp

def mock_covariance(p_mock, xi_mock):

    p_mean = np.mean( np.array(p_mock), axis = 0)
    xi_mean = np.mean( np.array(xi_mock), axis = 0)

    m1, m2 = np.mgrid[0:p_mean.size, 0:xi_mean.size]
    mock_covpxi = np.zeros(( p_mean.size, xi_mean.size ))

    #sys.stdout.flush()
    for i in range(len(p_mock)) :
        p = np.array(p_mock[i])
        x = np.array(xi_mock[i])
        mock_covpxi += ( p[m1] - p_mean[m1]) * (x[m2] - xi_mean[m2])
        print 'calculate cov {}/{}         \r'.format(i+1, len(p_mock)),

    #mock_covpxi = 0.5 * (mock_covpxi + mock_covpxi.T)
    mock_covpxi = 1./( len(p_mock) - 1 ) * mock_covpxi
    
    #if p_model.size == xi_model.size : mock_covpxi = (mock_covpxi + mock_covpxi.T)/2.
    return mock_covpxi


def fourier_tr_xi_mock(cosmo, l, kbin, p_mock):
    #p_mock_interp = scipy.interpolate.interp1d( kbin, p_mock )
    #kbin_fine = np.logspace(np.log10( kbin[0]), np.log10(kbin[-1]), num=20000 )
    #kbin_fine = kbin_fine[1:-1]
    #print kbin_fine[-1]
    #p_mock_fine = p_mock0_interp(kbin_fine)
    #xi_mock0 = cosmo.fourier_transform_kr(0, kbin_fine, p_mock_fine)
    xi_mock = []
    for i in range(len(p_mock)):
        #p_mock_interp = scipy.interpolate.interp1d( kbin, p_mock[i] )
        #print kbin_fine[0], kbin_fine[-1], kbin[-1], kbin[0]
        #p_mock_fine = p_mock_interp(kbin_fine)
        xi_mock.append(cosmo.fourier_transform_kr(l, kbin, p_mock[i]))
        #xi_mock.append(cosmo.fourier_transform_kr(0, cosmo.kbin, p_mock[i]))
        print 'fourier transform.. {:04}/{} \r'.format(i+1, len(p_mock)),
    xi_mock = np.array(xi_mock)
    return xi_mock


def _fourier_tr_xi_mock(cosmo, l, kbin, p_mock):
    #p_mock_interp = scipy.interpolate.interp1d( kbin, p_mock )
    kbin_fine = np.logspace(np.log10( kbin[0]), np.log10(kbin[-1]), num=20000 )
    kbin_fine = kbin_fine[1:-1]
    #p_mock_fine = p_mock0_interp(kbin_fine)
    #xi_mock0 = cosmo.fourier_transform_kr(0, kbin_fine, p_mock_fine)
    xi_mock = []
    for i in range(len(p_mock)):
        p_mock_interp = scipy.interpolate.interp1d( kbin, p_mock[i] )
        #print kbin_fine[0], kbin_fine[-1], kbin[-1], kbin[0]
        p_mock_fine = p_mock_interp(kbin_fine)
        xi_mock.append(cosmo.fourier_transform_kr(l, kbin_fine, p_mock_fine))
        #xi_mock.append(cosmo.fourier_transform_kr(0, cosmo.kbin, p_mock[i]))
        print 'fourier transform.. {:04}/{} \r'.format(i+1, len(p_mock)),
    xi_mock = np.array(xi_mock)
    return xi_mock


def generate_mocks( cosmo, l, kbin, p_model, covp_model, N_mock = 500 ):
    print 'generate mocks... size=', N_mock, '    '
    p_mock = np.array([np.random.normal(loc=p_model[i], scale=np.sqrt(covp_model.diagonal()[i]),
                                             size=N_mock) for i in range(kbin.size)]).T
    xi_mock = _fourier_tr_xi_mock(cosmo, l, kbin, p_mock)
    return p_mock, xi_mock


def _generate_mocks( cosmo, l, kbin, p_model, covp_model, N_mock = 500 ):
    print 'generate mocks... size=', N_mock
    p_mock = np.array([np.random.normal(loc=p_model[i], scale=np.sqrt(covp_model.diagonal()[i]),
                                             size=N_mock) for i in range(cosmo.kbin.size)]).T
    xi_mock = _fourier_tr_xi_mock(cosmo, l, kbin, p_mock)

    return p_mock, xi_mock
    
def save_mocks( cosmo, kbin, p_mock, rbin, xi_mock, header = '', dir = '../data_txt/mocks/'  ):
    if not os.path.exists(dir) : os.makedirs(dir) 
    DAT_p = np.column_stack(( p_mock ))
    DAT_xi= np.column_stack(( xi_mock))
    np.savetxt(dir + 'mocks_p.dat', DAT_p, header = header)
    np.savetxt(dir + 'mocks_xi.dat', DAT_xi,header = header)
    np.savetxt(dir + 'r.dat', rbin, header = header)
    np.savetxt(dir + 'k.dat', kbin, header = header)
    
def load_mocks_pxi( dir = '../data_txt/mocks/'  ):
    mocks_p = np.loadtxt(dir + 'mocks_p.dat')
    mocks_xi= np.loadtxt(dir + 'mocks_xi.dat')
    rcenter = np.loadtxt(dir + 'r.dat')
    kcenter = np.loadtxt(dir + 'k.dat')
    return kcenter, mocks_p.T, rcenter.T, mocks_xi.T

def load_mocks( dir = '../data_txt/mocks/', probe='p' ):

    if probe == 'p':
        filename = 'mocks_p.dat'
        bin = 'k.dat'
    elif probe == 'xi':
        filename = 'mocks_xi.dat'
        bin = 'r.dat'

    mocks = np.loadtxt(dir + filename)
    bincenter = np.loadtxt(dir + bin)
    return bincenter, mocks.T

def compute_data_vector(cosmo, l=0, b=2.0,f=0.74,s=3.5):
    cosmo.b = b
    cosmo.f = f
    cosmo.s = s
    datavp = cosmo.multipole_P(l)
    datavxi = cosmo.multipole_Xi(l)
    return [datavp, datavxi]


def interp_2d( x, y, z):
    f = scipy.interpolate.interp2d(x, y, z, kind='cubic')
    return f

# store theory data vector for different bias
def datavector_bias_2d_interp(cosmo):

    blist = np.linspace(0.5, 3.5, 51)
    print 'Calculate datavector p(b) and xi(b) for b = [{},{}]'.format(blist[0], blist[-1])
    datavlist = []
    datavlist_xi = []
    i=0
    for bb in blist:
        datavp, datavxi = compute_data_vector(cosmo, b=bb, s = cosmo.s)
        datavlist.append(datavp)
        datavlist_xi.append(datavxi)
        print '{:0.2f}  {}/{}                   \r'.format(bb, i+1, len(blist)),
        i+=1

    datavlist = np.array(datavlist)
    datavlist_xi = np.array(datavlist_xi)

    #print datavlist.shape, datavlist_xi.shape
    #datavlist_com = np.hstack((datavlist, datavlist_xi  ))
    print 'generate 2D interpolation table'
    datavp_interp = interp_2d( cosmo.kcenter_y, blist, datavlist )
    datavxi_interp = interp_2d( cosmo.rcenter[::-1], blist, datavlist_xi )

    cosmo.b = 2.0
    return datavp_interp, datavxi_interp

def getting_sigma_bs_theory( cosmo, b = None, cov = None, datavs = None, 
                     mockdatavs = None, p=False, kmin=None, kmax = None,
                     rmin = None, rmax = None):
    
    
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
    
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )
        
    mask = np.zeros(cov.shape[0], dtype=bool)
    if p : mask[idx_kmin:idx_kmax+1] = 1
    else : mask[idx_rmax:idx_rmin+1] = 1
        
    Nk = np.sum(mask)
    m1, m2 = np.mgrid[0:cov.shape[0],0:cov.shape[0]]
    mask_2d = mask[m1] * mask[m2]
           
    if p :print 'Nk ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax
    else : print 'Nr ', Nk, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    covinv = np.linalg.inv(cov[mask_2d].reshape(Nk,Nk))
    
    if p :dqdb = cosmo.dPdb0_interp(cosmo.kbin)
    else :dqdb = cosmo.dxidb
    dqdb = dqdb[mask]
    
    sigma_theory = 1./np.sqrt(np.dot( np.dot( dqdb, covinv), dqdb.T))
    print ' theory :', sigma_theory
    return sigma_theory




def getting_sigma_bs_com_theory( cosmo, b = None, cov = None, datavs = None, mockdatavs = None, p=False, 
                         kmin=None, kmax = None, rmin=None, rmax = None ):
    
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
        
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )
        
        
    maskp = np.zeros(cosmo.kbin.size, dtype=bool)
    maskp[idx_kmin:idx_kmax+1] = 1
    maskxi = np.zeros(cosmo.rcenter.size, dtype=bool)
    maskxi[idx_rmax:idx_rmin+1] = 1
    mask = np.hstack([maskp, maskxi])
    Nk = np.sum(mask)
    
    m1, m2 = np.mgrid[0:cov.shape[0],0:cov.shape[0]]
    mask_2d = mask[m1] * mask[m2]
    
    print 'N ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    #covinv = np.linalg.inv(cov[mask_2d].reshape(Nk, Nk))
    covp = cov[idx_kmin:idx_kmax+1, idx_kmin:idx_kmax+1]
    covxi = cov[cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1,\
                cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1]
    covpxi = cov[idx_kmin:idx_kmax+1,\
                cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1]
    
    #covinv = invert_covtot(covp ,covxi, covpxi)
    covtot = np.vstack(( np.hstack((covp, covpxi )), np.hstack((covpxi.T, covxi )) ))
    covinv = np.linalg.inv(covtot)
    #print covinv.diagonal
    #print 'covp', covp
    #print 'covxi', covxi
    print 'Warning : negative diagonal ', np.sum( covinv.diagonal() < 0)    
    
    dqdb = np.hstack((cosmo.dPdb0_interp(cosmo.kbin), cosmo.dxidb))[mask]    
    sigma_theory = 1./np.sqrt(np.dot( np.dot( dqdb, covinv), dqdb.T))
    print ' theory :', sigma_theory
    return sigma_theory


def getting_sigma_bs_diff_theory( cosmo, b = None, covp = None, covxi = None, 
                          datavsp = None, datavsxi = None, 
                          mockdatavsp = None, mockdatavsxi = None, p=False, 
                          kmin=None, kmax = None,
                          rmin = None, rmax = None):
 
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
        
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )

        
    maskp = np.zeros(cosmo.kbin.size, dtype=bool)
    maskp[idx_kmin:idx_kmax+1] = 1
    maskxi = np.zeros(cosmo.rcenter.size, dtype=bool)   
    maskxi[idx_rmax:idx_rmin+1] = 1
    Nk, Nr = np.sum(maskp), np.sum(maskxi) 
        
    m1, m2 = np.mgrid[0:covp.shape[0],0:covp.shape[0]]
    maskp_2d = maskp[m1] * maskp[m1].T
    m1, m2 = np.mgrid[0:covxi.shape[0],0:covxi.shape[0]]
    maskxi_2d = maskxi[m1] * maskxi[m1].T    
    
    print 'Nk ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax
    print 'Nr ', Nr, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    
    covinv_p = np.linalg.inv(covp[maskp_2d].reshape(Nk, Nk))
    covinv_xi = np.linalg.inv(covxi[maskxi_2d].reshape(Nr, Nr))
    
    dpdb = cosmo.dPdb0_interp(cosmo.kbin)[maskp]
    dxdb = cosmo.dxidb[maskxi]

    Fp = np.dot( np.dot(dpdb, covinv_p), dpdb.T)
    Fx = np.dot( np.dot(dxdb, covinv_xi), dxdb.T)
    
    sigma_theory = 1./np.sqrt((Fp+Fx))
    print ' theory :', sigma_theory
    return sigma_theory



def getting_sigma_bf_theory( cosmo, cov = None, datav = None, kmin=None, kmax = None):
    
    
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
        
    
    mask = np.zeros(cov.shape[0], dtype=bool)
    mask[idx_kmin:idx_kmax+1] = 1
        
    Nk = np.sum(mask)
    m1, m2 = np.mgrid[0:cov.shape[0],0:cov.shape[0]]
    mask_2d = mask[m1] * mask[m2]
    
    N_l = datav.shape[0]
    mask_datav = np.zeros((datav.shape), dtype=bool)
    for i in range(N_l): 
        mask_datav[i, idx_kmin:idx_kmax+1] = 1
           
    print ('Nk ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax)

    #covinv = np.linalg.inv(cov[mask_2d].reshape(Nk,Nk))

    covinv = np.zeros((Nk,Nk))
    covinv_diagonal =  1./cov[mask_2d].reshape(Nk,Nk).diagonal() 
    np.fill_diagonal(covinv, covinv_diagonal)

    dqdb = datav[mask_datav]
    dqdb = dqdb.reshape(N_l, dqdb.size/N_l)   


    fig, ax = plt.subplots()
    karray = cosmo.kcenter[mask]
    ax.plot(karray, covinv_diagonal*1e+6, 'k-', label='cov')
    for i in range(N_l):
        ax.plot(karray, dqdb[i,:] )
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()

    #dqdb = dqdb[0,:].reshape(1, dqdb.size/N_l )
    #print dqdb[0,:].shape
    fisher_theory = np.dot( np.dot( dqdb, covinv ), dqdb.T)
    sigma_theory = np.linalg.inv(fisher_theory) 

    #print ' theory :', sigma_theory
    return sigma_theory
        

def best_chisqr( covinv = None, datav=None, mockv = None, p=False):
    d_diff = (datav - mockv)
    if p : chisqr = np.sum(d_diff**2 * covinv.diagonal())
    else : chisqr = np.dot(np.dot( d_diff, covinv ), d_diff)
    return chisqr


def getting_sigma_bs( cosmo, b = None, cov = None, datavs = None, 
                     mockdatavs = None, p=False, kmin=None, kmax = None,
                     rmin = None, rmax = None):    
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kcenter_y.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kmin_y )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kmax_y )
    
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )
        
    mask = np.zeros(cov.shape[0], dtype=bool)
    if p : mask[idx_kmin:idx_kmax+1] = 1
    else : mask[idx_rmax:idx_rmin+1] = 1
        
    Nk = np.sum(mask)
    m1, m2 = np.mgrid[0:cov.shape[0],0:cov.shape[0]]
    mask_2d = mask[m1] * mask[m2]
           
    if p :print 'Nk ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax
    else : print 'Nr ', Nk, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    #print 'cov', cov[mask_2d].reshape(Nk,Nk)
    covinv = np.linalg.inv(cov[mask_2d].reshape(Nk,Nk))
    fig, ax = plt.subplots()
    
    bestfit_b = []
    chi2result = np.zeros(b.size)
    chi2result_m = []
    for i in range(mockdatavs.shape[0]):
    #for i in range(1):
        print '{}/{} \r'.format(i+1, mockdatavs.shape[0] ),
        for j in range(b.size) : 
            dv = datavs[j][mask]
            #mv = mockdatavs[:,i][mask]    
            mv = mockdatavs[i][mask] 
            chi2result[j] = best_chisqr( covinv = covinv, datav=dv, mockv = mv, p=p)
            #print dv.shape,  mv.shape, chi2result[j]
        maxind = np.argmin(chi2result)
        #print maxind
        bestfit_b.append(b[maxind])
        chi2result_m.append( chi2result )
        ax.plot(b, chi2result, color = 'grey', alpha = 0.2)
    
    if p : la = 'p only'
    else : la = 'xi only'
    ax.plot(b, chi2result, color = 'grey', alpha = 0.2, label = la)    
    ax.set_xlabel('b')
    ax.set_ylabel('chi2')
    ax.legend(loc='best')

    print ' sigma_b :', np.std(bestfit_b)
    
    
    
    if p :dqdb = cosmo.dPb0#cosmo.dPdb0_interp(cosmo.kbin)
    else :dqdb = cosmo.dxib0
    dqdb = dqdb[mask]
    
    sigma_theory = 1./np.sqrt(np.dot( np.dot( dqdb, covinv), dqdb.T))
    print ' theory :', sigma_theory
    
    return bestfit_b, chi2result_m



def getting_sigma_bs_com( cosmo, b = None, cov = None, datavs = None, mockdatavs = None, p=False, 
                         kmin=None, kmax = None, rmin=None, rmax = None ):
    
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
        
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )
        
        
    maskp = np.zeros(cosmo.kbin.size, dtype=bool)
    maskp[idx_kmin:idx_kmax+1] = 1
    maskxi = np.zeros(cosmo.rcenter.size, dtype=bool)
    maskxi[idx_rmax:idx_rmin+1] = 1
    mask = np.hstack([maskp, maskxi])
    Nk = np.sum(mask)
    
    m1, m2 = np.mgrid[0:cov.shape[0],0:cov.shape[0]]
    mask_2d = mask[m1] * mask[m2]
    
    print 'N ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    #covinv = np.linalg.inv(cov[mask_2d].reshape(Nk, Nk))
    covp = cov[idx_kmin:idx_kmax+1, idx_kmin:idx_kmax+1]
    covxi = cov[cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1,\
                cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1]
    covpxi = cov[idx_kmin:idx_kmax+1,\
                cosmo.kbin.size + idx_rmax:cosmo.kbin.size+idx_rmin+1]
    
    #covinv = invert_covtot(covp ,covxi, covpxi)
    covtot = np.vstack(( np.hstack((covp, covpxi )), np.hstack((covpxi.T, covxi )) ))
    covinv = np.linalg.inv(covtot)
    #print covinv.diagonal
    #print 'covp', covp
    #print 'covxi', covxi
    print 'Warning : negative diagonal ', np.sum( covinv.diagonal() < 0)
    
    fig, ax = plt.subplots()
    
    bestfit_b = []
    chi2result = np.zeros(b.size)
    chi2result_m = []
    for i in range(mockdatavs.shape[0]):
    #for i in range(100):
        print '{}/{} \r'.format(i+1, mockdatavs.shape[0] ),
        for j in range(b.size) : 
            dv = datavs[j][mask]
            mv = mockdatavs[i][mask]
            chi2result[j] = best_chisqr( covinv = covinv, datav=dv, mockv = mv)
        maxind = np.argmin(chi2result)
        bestfit_b.append(b[maxind])
        chi2result_m.append(chi2result)
        #print chi2result[maxind]
        ax.plot(b, chi2result, color = 'grey', alpha = 0.2)
    ax.plot(b, chi2result, color = 'grey', alpha = 0.2, label = 'p+xi+pxi') 
    ax.set_xlabel('b')
    ax.set_ylabel('chi2')  
    ax.set_title('combine') 
    ax.legend(loc='best')
    print ' sigma_b :', np.std(bestfit_b)
    
    
    dqdb = np.hstack((cosmo.dPdb0_interp(cosmo.kbin), cosmo.dxidb))[mask]    
    sigma_theory = 1./np.sqrt(np.dot( np.dot( dqdb, covinv), dqdb.T))
    print ' theory :', sigma_theory
    
    return bestfit_b, chi2result_m


def getting_sigma_bs_diff( cosmo, b = None, covp = None, covxi = None, 
                          datavsp = None, datavsxi = None, 
                          mockdatavsp = None, mockdatavsxi = None, p=False, 
                          kmin=None, kmax = None,
                          rmin = None, rmax = None):
 
    if kmin == None : 
        idx_kmin = 0
        idx_kmax = cosmo.kbin.size
    else : 
        idx_kmin = get_closest_index_in_data( kmin, cosmo.kbin )   
        idx_kmax = get_closest_index_in_data( kmax, cosmo.kbin )
        
    if rmin == None : 
        idx_rmin = cosmo.rmax.size
        idx_rmax = 0
    else : 
        idx_rmin = get_closest_index_in_data( rmin, cosmo.rmin )   
        idx_rmax = get_closest_index_in_data( rmax, cosmo.rmax )

        
    maskp = np.zeros(cosmo.kbin.size, dtype=bool)
    maskp[idx_kmin:idx_kmax+1] = 1
    maskxi = np.zeros(cosmo.rcenter.size, dtype=bool)   
    maskxi[idx_rmax:idx_rmin+1] = 1
    Nk, Nr = np.sum(maskp), np.sum(maskxi) 
        
    m1, m2 = np.mgrid[0:covp.shape[0],0:covp.shape[0]]
    maskp_2d = maskp[m1] * maskp[m1].T
    m1, m2 = np.mgrid[0:covxi.shape[0],0:covxi.shape[0]]
    maskxi_2d = maskxi[m1] * maskxi[m1].T    
    
    print 'Nk ', Nk, ' kmin', idx_kmin, ' kmax', idx_kmax
    print 'Nr ', Nr, ' rmin', idx_rmin, ' rmax', idx_rmax
    
    
    covinv_p = np.linalg.inv(covp[maskp_2d].reshape(Nk, Nk))
    covinv_xi = np.linalg.inv(covxi[maskxi_2d].reshape(Nr, Nr))
    
    bestfit_b = []
    chi2result = np.zeros(b.size)
    chi2result1 = np.zeros(b.size)
    chi2result2 = np.zeros(b.size)
    
    chi2result_m = []
    fig, ax = plt.subplots()
    for i in range(mockdatavsp.shape[0]):
    #for i in range(10):
        print '{}/{} \r'.format(i+1, mockdatavsp.shape[0] ),
        for j in range(b.size) : 
            
            dv = datavsp[j][maskp]
            mv = mockdatavsp[i][maskp]
            dvxi = datavsxi[j][maskxi]
            mvxi = mockdatavsxi[i][maskxi]
            chi1 = best_chisqr( covinv = covinv_p, datav=dv, mockv = mv, p=True)
            chi2 = best_chisqr( covinv = covinv_xi, datav=dvxi, mockv = mvxi)
            chi2result[j] = chi1+chi2
            chi2result1[j] = chi1
            chi2result2[j] = chi2
        maxind = np.argmin(chi2result)
        bestfit_b.append(b[maxind])
        chi2result_m.append(chi2result)
        #print chi2result[maxind]
        ax.plot(b, chi2result, color = 'grey', alpha = 0.2)
        ax.plot(b, chi2result1, color = 'blue', alpha = 0.2)
        ax.plot(b, chi2result2, color = 'red', alpha = 0.2)
 

    ax.plot(b, chi2result, color = 'grey', alpha = 0.2, label = 'p+xi')
    ax.plot(b, chi2result1, color = 'blue', alpha = 0.2, label = 'p')
    ax.plot(b, chi2result2, color = 'red', alpha = 0.2, label = 'xi')


    ax.set_xlabel('b')
    ax.set_ylabel('chi2')  
    ax.set_title('diff') 
    ax.legend(loc='best')
        #stop
    
    print ' sigma_b :', np.std(bestfit_b)
    
    dpdb = cosmo.dPdb0_interp(cosmo.kbin)[maskp]
    dxdb = cosmo.dxidb[maskxi]

    Fp = np.dot( np.dot(dpdb, covinv_p), dpdb.T)
    Fx = np.dot( np.dot(dxdb, covinv_xi), dxdb.T)
    
    sigma_theory = 1./np.sqrt((Fp+Fx))
    print ' theory :', sigma_theory
    
    return bestfit_b, chi2result_m


def cross_b(bestfit_b_p, bestfit_b_xi):
    """
    cross covariance of bestfit b_p and bestfit b_xi
    """
    
    bestfit_b_p = np.array(bestfit_b_p)
    bestfit_b_xi = np.array(bestfit_b_xi)
    b_true = 2.0
    N = len(bestfit_b_p)
    return np.sum((bestfit_b_p - b_true)*(bestfit_b_xi - b_true)) * 1./(N-1)


def cross_params(bestfit_params_p, bestfit_params_xi):
    """
    cross covariance of bestfit b_p and bestfit b_xi
    """
    mean_params_p = np.mean( bestfit_params_p, axis = 0 )
    mean_params_xi = np.mean( bestfit_params_xi, axis = 0 )

    N_mocks, N_params = bestfit_params_p.shape
    cross_cov = np.zeros(( N_params, N_params ))
    M1, M2 = np.mgrid[0:N_params, 0:N_params]

    factor = 1./( N_mocks - 1. )

    for pi in range( N_mocks ):
        for xi in range( N_mocks ):
            cross_cov += (bestfit_params_p[pi,:][M1] - mean_params_p[M1]) *  (bestfit_params_xi[xi,:][M2] - mean_params_xi[M2] )

    return factor * cross_cov 
    #bestfit_b_p = np.array(bestfit_b_p)
    #bestfit_b_xi = np.array(bestfit_b_xi)
    #b_true = 2.0
    #N = len(bestfit_b_p)
    #return np.sum((bestfit_b_p - b_true)*(bestfit_b_xi - b_true)) * 1./(N-1)


def stack3x3(a=None, b=None, c=None):
    
    A = np.hstack(a)
    B = np.hstack(b)
    C = np.hstack(c)
    #print a[0].shape, A.shape
    return np.vstack([A,B,C])

def stack2x2(a=None, b=None):
    A = np.hstack(a)
    B = np.hstack(b)
    return np.vstack([A,B])

def combine_sigma(bestfit_params_p, bestfit_params_xi, re=True):
    
    if re:
        print 'reordering mock xi '
        bestfit_params_xi = np.vstack((bestfit_params_xi[1:,], bestfit_params_xi[0,:]))

    covp_mock = mock_covariance(bestfit_params_p, bestfit_params_p)
    #covpxi_bf_mock = mock_covariance(bestfit_params_p, bestfit_params_xi)
    covxi_mock = mock_covariance(bestfit_params_xi, bestfit_params_xi)
    bestfit_params = np.hstack([bestfit_params_p, bestfit_params_xi])
    covtot_mock = mock_covariance(bestfit_params, bestfit_params)
    #cross_cov = cross_params(bestfit_params_p, bestfit_params_xi)

    
    sigma_p = np.sqrt(covp_mock.diagonal())
    sigma_xi = np.sqrt(covxi_mock.diagonal())


    Fdiff = np.linalg.inv( covp_mock ) + np.linalg.inv( covxi_mock )
    covdiff = np.linalg.inv(Fdiff)
    sig_diff = np.sqrt(covdiff.diagonal())


    #covtot_mock = stack2x2( a = [covp_bf_mock, cross_cov], b = [cross_cov.T, covxi_bf_mock] )
    F_covtot_mock = np.linalg.inv(covtot_mock)

    projection_vector_ = np.zeros(( sigma_p.size, sigma_p.size ))
    np.fill_diagonal(projection_vector_, 1)
    projection_vector = np.vstack(( projection_vector_, projection_vector_ ))

    F_covtot_mock_projected = np.dot( np.dot( projection_vector.T, F_covtot_mock  ), projection_vector )
    C_covtot_projected = np.linalg.inv(F_covtot_mock_projected)

    sig_com = np.sqrt( C_covtot_projected.diagonal() )


    #F_covtot_theory_projected = np.dot( np.dot( projection_vector.T, np.linalg.inv(covtot_bf_theory)  ), projection_vector )
    #C_covtot_theory_projected = np.linalg.inv(F_covtot_theory_projected)
    
    #print 'cov matrix = \n |{:0.10f}   {:0.10f}| \n |{:0.10f}   {:0.10f}|'\
    #.format(cov[0,0], cov[0,1], cov[1,0], cov[1,1])
    
    print '\nsigma_p       :', sigma_p
    print 'sigma_xi      :', sigma_xi
    print 'sigma_combin  :', sig_com
    print 'sigma_diff    :', sig_diff

    #print '\nsigma_com_theory:', sig_com_theory
    dic = {}
    dic['mean_p'] = np.mean(bestfit_params_p, axis=0)
    dic['mean_Xi'] = np.mean(bestfit_params_xi, axis=0)
    dic['mean_tot'] = np.mean( np.vstack([bestfit_params_p,bestfit_params_xi]),axis=0)	
    dic['cov_params_p_direct'] = covp_mock
    dic['cov_params_Xi_direct'] = covxi_mock
    dic['cov_params_tot_direct'] = C_covtot_projected
    dic['cov_params_diff_direct'] = covdiff

    dic['sigma_params_p_direct'] = sigma_p
    dic['sigma_params_Xi_direct'] = sigma_xi
    dic['sigma_params_tot_direct'] = sig_com
    dic['sigma_params_diff_direct'] = sig_diff

    return dic #covp_mock, covxi_mock, C_covtot_projected, covdiff

