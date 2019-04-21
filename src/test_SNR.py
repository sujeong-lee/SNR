"""
This program calculates the cumulative signal to noise of two kinds of bandpower P and P_{xi}.
P_{xi} is obtained from correlation function Xi by Fourier transform.


* correlation function

for monopole,

Xi_0 (r) = Integral P_0(k) j0(kr) k^2 dk /(2 \pi^2 )


* Cumulative signal to noise


     k
    ---
    \
    /    P(k) [Cov P(k)]^-1 P(k)
    ---
    k_min



* Cov P_{xi}


    d Xi              d Xi
=   ---- [Cov P]^(-1) ----
    d P               d P




USAGE
------

Need multiprocessing module, f2py module.
* multiprocessing: https://docs.python.org/2/library/multiprocessing.html
* f2py: http://docs.scipy.org/doc/numpy-dev/f2py/

Either Linear_covariance or RSD_covariance class should be called first.
These classes take the initial setting parameters and define scales and spacings of models.
For details, see Class code (error_analysis_class.py).

KMIN and KMAX represent the beginning and end points of the Fourier integral.
Can be set to the smallest and the biggest k values of your data.

for test, set
RMIN = .1
RMAX = 200.
kmin = KMIN
kmax = KMAX

and see if two lines are agreed.

for RSD+BAO scale, use
BAO+RSD scale : RMIN=24, RMAX=152, kmin=0.01, kmax=0.2
RMIN = 24.
RMAX = 152.
kmin = 0.01
kmax = 0.2

for BAO only scale, use
RMIN = 29.
RMAX = 200.
kmin = 0.02
kmax = 0.3

MatterPower() load data from the input file.
multipole_P_band_all() generate bandpowers. For the Linear case, use Shell_avg_band() instead.

The next three functions create covariance matrices and derivatives for Fisher matrix calculation.
derivative_Xi_band_all()
RSDband_covariance_PP_all()
RSDband_covariance_Xi_all()

SNR_multiprocessing() does parallel calculation to run the function Cumulative_SNR_loop, that calculates the cumulative signal to noise at each k bin.

Detailed descriptions are included in the class code.

"""


import time, datetime
import numpy as np
from numpy import zeros, sqrt, pi, vectorize
from numpy.linalg import pinv as inv
from multiprocessing import Process, Queue
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from error_analysis_class import *
from noshellavg import *

def DataSave(RSDPower, kmin, kmax, suffix = None):

    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax )
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]

    #matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    
    #Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    #matrices2Xi = [RSDPower.dxip0, Xizeros,Xizeros,Xizeros,RSDPower.dxip2,Xizeros,Xizeros,Xizeros,RSDPower.dxip4]



    l = len(RSDPower.kcenter)
    karray = np.array([RSDPower.kcenter[kcut_min:kcut_max+1],RSDPower.kcenter[kcut_min:kcut_max+1],RSDPower.kcenter[kcut_min:kcut_max+1]]).ravel()
    
    #print l, len(karray),
    # F_bandpower from P
    
    Cov_bandpower_PP = CombineCovariance3(l, matricesPP)
    #print Cov_bandpower_PP.shape
    
    
    DAT1 = np.vstack(( karray, Cov_bandpower_PP ))
    
    # GET full C_Xi
    #l_r = len(RSDPower.rcenter)
    #C_matrix3 = CombineCovariance3(l_r, matricesXi)
    
    # F_bandpower from Xi
    """
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    Cov_bandpower_Xi = inv( Fisher_bandpower_Xi)
    np.allclose(Fisher_bandpower_Xi, np.dot(Fisher_bandpower_Xi, np.dot(Cov_bandpower_Xi, Fisher_bandpower_Xi)))
    """
    """
    fig,(ax, ax2) = plt.subplots(1,2, figsize = (10,5))
    im = ax.imshow(np.log10(Fisher_bandpower_Xi))
    im2 = ax2.imshow(np.log10(Cov_bandpower_Xi))
    cbar = fig.colorbar(im, ax = ax, fraction=0.046,pad=0.04)
    cbar.set_label('log10(F)')
    cbar2 = fig.colorbar(im2, ax = ax2, fraction=0.046,pad=0.04)
    cbar2.set_label('log10(C)')
    
    ax.set_title('Fisher_bandpower from Xi')
    ax2.set_title('Cov_bandpower from Xi')
    fig.savefig('plots/test2')
    """
    #DAT2 = np.vstack(( karray, Cov_bandpower_Xi ))
    #DAT3 = np.vstack(( karray, Fisher_bandpower_Xi ))

    # data vector P
    #data_Vec = np.array([RSDPower.multipole_bandpower0[0:l+1], RSDPower.multipole_bandpower2[0:l+1], RSDPower.multipole_bandpower4[0:l+1]]).reshape(1,3 * (l+1))
    DATVEC = np.column_stack(( RSDPower.kcenter[kcut_min:kcut_max+1], RSDPower.multipole_bandpower0[kcut_min:kcut_max+1], RSDPower.multipole_bandpower2[kcut_min:kcut_max+1], RSDPower.multipole_bandpower4[kcut_min:kcut_max+1] ))
    
    
    
    np.savetxt('covP'+suffix+'.txt', DAT1, delimiter = ' ', header = '#\nCovPP', comments = '# first column is k'  )
    #np.savetxt('covXi'+suffix+'.txt', DAT2, delimiter = ' ', header = '#\nCovPP from Xi', comments = '# first column is k'  )
    #np.savetxt('FisherXi'+suffix+'.txt', DAT3, delimiter = ' ', header = '#\nFisherPP from Xi', comments = '# first column is k'  )
    np.savetxt('dataVector'+suffix+'.txt', DATVEC , delimiter = ' ', header = ' k       P0        P2       P4', comments = '#'  )
    print 'data saved'


def ReidResult(RSDPower, rmin, rmax, kmin, kmax):
    
    from noshellavg import CombineDevXi
    
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    rcut_min = get_closest_index_in_data( rmin, RSDPower.rmin )
    rcut_max = get_closest_index_in_data( rmax, RSDPower.rmax )
    
    print rcut_min, rcut_max
    
    matricesXi = [RSDPower.covariance00[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance22[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  np.transpose(RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance44[rcut_max:rcut_min+1,rcut_max:rcut_min+1]]

    """
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    """
    matrices2Xi = [RSDPower.dxib0[rcut_max:rcut_min+1],\
                   RSDPower.dxib2[rcut_max:rcut_min+1],\
                   RSDPower.dxib4[rcut_max:rcut_min+1],\
                   RSDPower.dxif0[rcut_max:rcut_min+1],\
                   RSDPower.dxif2[rcut_max:rcut_min+1],\
                   RSDPower.dxif4[rcut_max:rcut_min+1],\
                   RSDPower.dxis0[rcut_max:rcut_min+1],\
                   RSDPower.dxis2[rcut_max:rcut_min+1],\
                   RSDPower.dxis4[rcut_max:rcut_min+1]]
    
    
    
    Fisherb = []
    Fisherf = []
    Fisherb_det = []
    Fisherf_det = []
    rrlist = []
    for l in range(RSDPower.rcenter[rcut_max:rcut_min+1].size):
        C_matrix3 = CombineCovariance3(l, matricesXi)
        FisherXi = inv(C_matrix3)
        DataVec, DataVec2 = CombineDevXi(l, matrices2Xi)
        Fisher = np.dot( np.dot( DataVec, FisherXi), DataVec.T )
        Cov = inv(Fisher)
        Cov_det = inv(Fisher[0:2, 0:2])
        Fisherb.append(np.sqrt(Cov[0,0]))
        Fisherf.append(np.sqrt(Cov[1,1]))
        Fisherb_det.append(np.sqrt(Cov_det[0,0]))
        Fisherf_det.append(np.sqrt(Cov_det[1,1]))
        rrlist.append( RSDPower.rcenter[rcut_max:rcut_min+1][l])

    rrlist = np.array(rrlist).ravel()
    Fisherb = np.array(Fisherb).ravel()/RSDPower.b
    Fisherf = np.array(Fisherf).ravel()/RSDPower.f
    Fisherb_det = np.array(Fisherb_det).ravel()/RSDPower.b
    Fisherf_det = np.array(Fisherf_det).ravel()/RSDPower.f

    fig, (ax, ax2) = plt.subplots(2,1)

    ax.semilogy( rrlist, Fisherb , 'r--', label = 's marginalized')
    ax2.plot( rrlist, Fisherf, 'b--', label = 's marginalized')
    ax.semilogy( rrlist, Fisherb_det , 'r-', label = 's determined')
    ax2.plot( rrlist, Fisherf_det, 'b-', label = 's determined')
    ax.set_xlim(0, 60)
    ax.set_ylim(0.0005, 0.05)
    ax.set_ylabel('b')
    ax.set_xlabel('r (Mpc/h)')
    ax2.set_xlim(0,60)
    ax2.set_ylim(0.00, 0.07)
    ax2.set_ylabel('f')
    ax2.set_xlabel('r (Mpc/h)')

    ax.legend(loc = 'best')
    ax2.legend(loc='best')

    return rrlist, Fisherb, Fisherf, Fisherb_det, Fisherf_det




def blockwise( matrix ):

    # F_upleft = a - b d^-1 c

    s = matrix[0:-1, 0:-1]
    w = matrix[-1:, 0:-1]
    #wT = matrix[0:-1, -1:]
    d = matrix[-1, -1]

    if d != 0.0 : F_upleft = s - np.dot( w.T, w )/d
    if d == 0.0 : F_upleft = s
    return F_upleft


def blockwise3x3( matrix ):

    # F_upleft = a - b d^-1 c

    s = matrix[0:-3, 0:-3]
    w = matrix[-3:, 0:-3]
    #wT = matrix[0:-1, -1:]
    d = matrix[-3:, -3:]
    
    #F_upleft = s - np.dot( w.T, w )/d
    F_upleft = s - np.dot( np.dot( w.T, inv(d)), w)
    return F_upleft


def blockwiseInversion( matrix, cutInd ):

    from numpy.linalg import inv
    
    a = matrix[0:cutInd+1, 0:cutInd+1]
    b = matrix[0:cutInd+1, cutInd+1:]
    c = b.T #matrix[cutInd+1:, 0:cutInd+1]
    d = matrix[cutInd+1:, cutInd+1:]
    ia = inv(a)

    Fd = inv( d - np.dot( np.dot( c, ia ), b) )
    Fc = - np.dot( np.dot( Fd, c), ia)
    Fb = - np.dot( np.dot( ia, b ), Fd )
    Fa = ia + np.dot( np.dot (np.dot( np.dot( ia, b), Fd ), c), ia)
    
    F = np.vstack(( np.hstack(( Fa, Fb )), np.hstack(( Fc, Fd )) ))
    return F



def _reordering( matrix, l=1 ):

    #if cut is None : cut = RSDPower.kcenter_y.size  #len(RSDPower.kcenter)
    
    cut = matrix.shape[0]/l
    part00 = matrix[0:cut, 0:cut]
    if l >= 2 : 
        part02 = matrix[0:cut, cut:2*cut]
        part22 = matrix[cut:2*cut, cut:2*cut]
    if l == 3 :
        part04 = matrix[0:cut, 2*cut:]
        part24 = matrix[cut:2*cut, 2*cut:]
        part44 = matrix[2*cut:, 2*cut:]

    ReorderedF = np.zeros(( matrix.shape ))
    
    #if l == 1 : ReorderedF = part00
    #else : 
    for i in range(cut):
        for j in range(cut):
            ReorderedF[l*i, l*j] = part00[i,j]
            if l >=2 :
                ReorderedF[l*i, l*j+1] = part02[i,j]
                ReorderedF[l*i+1, l*j] = part02[j,i]
                ReorderedF[l*i+1, l*j+1] = part22[i,j]
            if l == 3:
                ReorderedF[l*i, 3*j+2] = part04[i,j]
                ReorderedF[l*i+2, l*j] = part04[j,i]
                ReorderedF[l*i+1, l*j+2] = part24[i,j]
                ReorderedF[l*i+2, l*j+1] = part24[j,i]
                ReorderedF[l*i+2, l*j+2] = part44[i,j]

    return ReorderedF


def reordering( RSDPower, matrix ):

    #if cut is None : cut = RSDPower.kcenter_y.size  #len(RSDPower.kcenter)
    
    cut = matrix[:,0].size/3
    part00 = matrix[0:cut, 0:cut]
    part02 = matrix[0:cut, cut:2*cut]
    part04 = matrix[0:cut, 2*cut:]
    part22 = matrix[cut:2*cut, cut:2*cut]
    part24 = matrix[cut:2*cut, 2*cut:]
    part44 = matrix[2*cut:, 2*cut:]

    ReorderedF = np.zeros(( matrix.shape ))
    
    ReorderedP = np.zeros(3 * cut)
    
    ind = np.arange(0,3*cut, 3)
    ind2 = ind + 1
    ind3 = ind + 2
    
    ReorderedP[ind] = RSDPower.multipole_bandpower0
    ReorderedP[ind2] = RSDPower.multipole_bandpower2
    ReorderedP[ind3] = RSDPower.multipole_bandpower4
    
    for i in range(cut):
        for j in range(cut):
            ReorderedF[3*i, 3*j] = part00[i,j]
            ReorderedF[3*i, 3*j+1] = part02[i,j]
            ReorderedF[3*i+1, 3*j] = part02[j,i]
            ReorderedF[3*i, 3*j+2] = part04[i,j]
            ReorderedF[3*i+2, 3*j] = part04[j,i]
            ReorderedF[3*i+1, 3*j+1] = part22[i,j]
            ReorderedF[3*i+1, 3*j+2] = part24[i,j]
            ReorderedF[3*i+2, 3*j+1] = part24[j,i]
            ReorderedF[3*i+2, 3*j+2] = part44[i,j]

    return ReorderedF, ReorderedP



    
def reorderingVector( vector, l = 1 ):
    
    #if len(vector) > 1: vector = np.hstack(vector)
        
    try : 
        nx, ny = vector.shape
        ReorderedP = np.zeros((vector.shape))
        ind = np.arange(0, ny, l)

        Ny = ny/l
        for j in range(Ny):
            for i in range(nx):
                ReorderedP[i,:][ind] = vector[i][:Ny]
                if l >= 2 :
                    ReorderedP[i,:][ind+1] = vector[i][Ny:Ny*2]
                if l == 3 :
                    ReorderedP[i,:][ind+2] = vector[i][2*Ny:3*Ny]

    except : 
 
        Nx = vector.size/l
        ReorderedP = np.zeros((vector.shape))
        ind = np.arange(0,vector.size, l)

        Nx = vector.size/l
        ReorderedP[ind] = vector[:Nx]
        if l >= 2 : ReorderedP[ind+1] = vector[Nx:Nx*2]
        if l == 3 : ReorderedP[ind+2] = vector[Nx*2:Nx*3]
    
    return ReorderedP



def _reorderingVector( vector ):
    
    n = len(vector)    
    
    try :     
        
        cut = len(vector[0][0,:])
        ReorderedP = np.zeros(( vector[0][:, 0].size ,n * cut))
        
        ind = np.arange(0,n * cut, n)
        #ind2 = ind + 1
        #ind3 = ind + 2
        
        for i in range( vector[0][:, 0].size ):
            for j in range(n):
                IND = ind + j
                ReorderedP[i,:][IND] = vector[j][i,:]
            #ReorderedP[i,:][ind] = vector[0][i,:]
            #ReorderedP[i,:][ind2] = vector[1][i,:]
            #ReorderedP[i,:][ind3] = vector[2][i,:]
    
    
    except IndexError:
        cut = vector[0].size
        vector = np.array(vector)
        ReorderedP = np.zeros(vector.size)
        
        ind = np.arange(0, n * cut, n)
        #ind2 = ind + 1
        #ind3 = ind + 2

        for j in range(n):
            IND = ind + j
            ReorderedP[IND] = vector[j]


            #ReorderedP[ind] = vector[0]
            #ReorderedP[ind2] = vector[1]
            #ReorderedP[ind3] = vector[2]

    return ReorderedP
    



def convergence_Xi(RSDPower, rmin, rmax, kmin, kmax):
    
    from noshellavg import get_closest_index_in_data
    
    if kmin is None :
        kcut_min = 0
        kcut_max = RSDPower.dxip0[:,0].size
    else:
        kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
        kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    rcut_min = get_closest_index_in_data( rmin, RSDPower.rmin )
    rcut_max = get_closest_index_in_data( rmax, RSDPower.rmax )

    
    matricesXi = [RSDPower.covariance00[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance22[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  np.transpose(RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance44[rcut_max:rcut_min+1,rcut_max:rcut_min+1]]
    
    Xizeros = np.zeros(RSDPower.dxip0.shape)[kcut_min:kcut_max+1,rcut_max:rcut_min+1]

    
    matrices2Xi = [RSDPower.dxip0[kcut_min:kcut_max+1:,rcut_max:rcut_min+1]
                   , Xizeros
                   , Xizeros
                   , Xizeros
                   , RSDPower.dxip2[kcut_min:kcut_max+1:,rcut_max:rcut_min+1]
                   , Xizeros
                   , Xizeros
                   , Xizeros
                   , RSDPower.dxip4[kcut_min:kcut_max+1:,rcut_max:rcut_min+1]]
    
    multipole_P0, multipole_P2, multipole_P4 = RSDPower.multipole_bandpower0, RSDPower.multipole_bandpower2, RSDPower.multipole_bandpower4
    

    # GET full C_Xi
    l_r = RSDPower.rcenter[rcut_max:rcut_min+1].size
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    print 'sum of Fisher', np.sum(inv(C_matrix3))
    
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)  # Here, cut out small and large k

    # F_bandpower from Xi
    
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    print 'sum of Fisher', np.sum(Fisher_bandpower_Xi)
    #print 'diagonal comp', Fisher_bandpower_Xi.diagonal()
    
    cut = kcut_max - kcut_min
    matrix1, matrix2 = np.mgrid[0:cut, 0:cut]
    P1 = RSDPower.multipole_bandpower0[matrix1]
    
    #cut = RSDPower.kcenter_y.size
    #part00 = blockwise(Fisher_bandpower_Xi, cut)
    
    """
    Cov_bandpower_Xi = inv( Fisher_bandpower_Xi ) #, rcond = 1e-10 )
    #cut = RSDPower.kcenter_y.size  #len(RSDPower.kcenter)
    part00 = Cov_bandpower_Xi[0:cut, 0:cut]
    part02 = Cov_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Cov_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Cov_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Cov_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Cov_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]

    part_list = [ part00, part02, part04, np.transpose(part02), part22, part24, np.transpose(part04), np.transpose(part24), part44]
    
    Cov_bandpower_Xi_combine = CombineCovariance3(cut, part_list)
    part_Fisher_bandpower_Xi = inv( Cov_bandpower_Xi_combine )
    """
    # blockwise method
    F, P = reordering( RSDPower, Fisher_bandpower_Xi)

    SNRlist2 = []
    SNR = np.dot( np.dot(P, F), P.T )
    SNRlist2.append(SNR)
    for j in range(1, cut):
        P = P[:-3]
        for i in range(0,3):
            F = blockwise( F )

        SNR = np.dot( np.dot(P, F), P.T )
        SNRlist2.append(SNR)

    #SNRlist2 = np.array(SNRlist2).ravel()
    #SNRlist2 = [ SNRlist2[3*i] for i in range(RSDPower.kcenter_y.size) ]
    SNRlist2 = np.array(SNRlist2[::-1]).ravel()

    #return RSDPower.kcenter_y, SNRlist, SNRlist2

    """
    SNR_Xi_list = []
    kklist = []

    for l in range(0, cut-kcut_min, 1):
        kklist.append(RSDPower.kcenter_y[kcut_min + l])
        Cov_bandpower_Xi_combine = CombineCovariance3(l, part_list)
        
        part_Fisher_bandpower_Xi = inv( Cov_bandpower_Xi_combine ) #, rcond = 1e-15)
        data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).ravel()

        SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), data_Vec.T)
        SNR_Xi_list.append(SNR_Xi)

    #diagonal = Fisher_bandpower_Xi.diagonal() / data_Vec**2
    SNR_Xi_list = np.array(SNR_Xi_list).ravel()
    """
    return RSDPower.kcenter_y[kcut_min:kcut_max + 1], SNRlist2



def convergence_P(RSDPower, kmin, kmax):
    """
    from linear import Covariance_PP, Pmultipole, covariance_Xi, derivative_Xi
    cov00 = Covariance_PP(0,0)
    P0 = Pmultipole(0)
    covXi00 = covariance_Xi(0,0)
    dxip0 = derivative_Xi(0)
    FisherXi = np.dot( np.dot(dxip0.T, inv(covXi00)), dxip0 )
    covXi = inv(FisherXi)
    SNR_PP_list = []
    SNR_Xi_list = []
    for l in range(1, P0.size/10):
        part_Fisher_bandpower_PP = inv(cov00[0:l, 0:l])
        part_Fisher_bandpower_Xi = inv(covXi[0:l, 0:l])
        data_Vec = np.array(P0[0:l])
        SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
        SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), np.transpose(data_Vec))
        SNR_PP_list.append(SNR_PP)
        SNR_Xi_list.append(SNR_Xi)
        print SNR_PP, SNR_Xi

    SNR_PP_list = np.array(SNR_PP_list).ravel()
    SNR_Xi_list = np.array(SNR_Xi_list).ravel()
    """
    #from linear import Covariance_PP, Pmultipole
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    #l = kcut_max - kcut_min

    multipole_P0, multipole_P2, multipole_P4 = RSDPower.multipole_bandpower0[kcut_min:kcut_max+1], RSDPower.multipole_bandpower2[kcut_min:kcut_max+1], RSDPower.multipole_bandpower4[kcut_min:kcut_max+1]
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
       
       
    l = kcut_max - kcut_min + 1
    cov = CombineCovariance3(l, matricesPP)
    part_Fisher_bandpower_PP = inv(cov)
    #cov = np.ma.masked_where( cov == 0, cov )
    #part_Fisher_bandpower_PP = np.ma.masked_where( part_Fisher_bandpower_PP == 0, part_Fisher_bandpower_PP )
    """
    fig, (ax, ax2) = plt.subplots(1,2)
    im = ax.imshow(part_Fisher_bandpower_PP )
    im2 = ax2.imshow(cov )
    ax.set_title('Fisher')
    ax2.set_title('Cov')
    fig.colorbar(im, ax=ax)
    fig.colorbar(im2, ax = ax2)
    """
    print 'sum of Fisher',  np.sum(part_Fisher_bandpower_PP)

    SNR_PP_list = []
    kklist = []
    #first compo
    """
    part_Fisher_bandpower_PP = inv(np.array([ matrix[0, 0] for matrix in matricesPP ]).reshape(3,3))
    data_Vec = np.array([multipole_P0[0], multipole_P2[0], multipole_P4[0]]).ravel()
    SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
    SNR_PP_list.append(SNR_PP)
    """

    for l in range(0, kcut_max - kcut_min + 1, 1):
        #kklist.append(RSDPower.kcenter_y[l])
        kklist.append(RSDPower.kcenter_y[kcut_min + l])
        part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
        data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).ravel()
        SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
        #print SNR_PP
        SNR_PP_list.append(SNR_PP)
    
    """
    SNR_PP_list_1 = []
    for l in range((kcut_max - kcut_min + 1)):
        Fisher = inv(CombineCovariance3(l, matricesPP))
        part_Fisher_bandpower_PP = np.array([ Fisher[l,l],  Fisher[l,2*l + 1],  Fisher[l,3*l + 2],
                                    Fisher[2*l+1,l],  Fisher[2*l+1,2*l + 1],  Fisher[2*l+1,3*l + 2],
                                    Fisher[3*l+2,l],  Fisher[3*l+2,2*l + 1],  Fisher[3*l+2,3*l + 2]]).reshape(3,3)
        data_Vec = np.array([multipole_P0[l], multipole_P2[l], multipole_P4[l]]).ravel()
        SNR_PP_1 = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), data_Vec)

        SNR_PP_list_1.append(SNR_PP_1)
    """
    """
    cov00, cov02, cov04, cov22, cov24, cov44 = Covariance_PP(0,0), Covariance_PP(0,2), Covariance_PP(0,4), Covariance_PP(2,2), Covariance_PP(2,4), Covariance_PP(4,4)
    
    print 'cov'
    matricesPP = [ cov00, cov02, cov04, cov02, cov22, cov24, cov04, cov24, cov44 ]
    P0, P2, P4 = Pmultipole(0), Pmultipole(2), Pmultipole(4)
    
    print 'multipole'
    #l = len(cov00)
    
    SNR_PP_list = []
    for l in range(P0.size/2):
        part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
        data_Vec = np.array([P0[0:l+1], P2[0:l+1], P4[0:l+1]]).reshape(1,3 * P0[0:l+1].size)
        SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
        SNR_PP_list.append(SNR_PP)
    SNR_PP_list = np.array(SNR_PP_list).ravel()
    """
    
    return np.array(kklist), np.array(SNR_PP_list).ravel()



def BinAvgSNR_P(RSDPower, kmin, kmax ):

    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )

    multipole_P0, multipole_P2, multipole_P4 = RSDPower.multipole_bandpower0[kcut_min:kcut_max+1], RSDPower.multipole_bandpower2[kcut_min:kcut_max+1], RSDPower.multipole_bandpower4[kcut_min:kcut_max+1]
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]

    l = kcut_max - kcut_min + 1
    cov = CombineCovariance3(l, matricesPP)
    part_Fisher_bandpower_PP = inv(cov)

    print 'sum of Fisher',  np.sum(part_Fisher_bandpower_PP)
    
    SNR_PP_list = []
    kklist = []

    for l in range(0, kcut_max - kcut_min + 1, 1):
        kklist.append(RSDPower.kcenter_y[kcut_min + l])
        part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
        # fill zero except last column and row
        if l == 0 : pass
        else :
            mask = np.zeros(part_Fisher_bandpower_PP.shape, dtype=bool)
            mask[l,:],  mask[:,l], mask[2*l+1,:],  mask[:,2*l+1], mask[3*l+2,:],  mask[:,3*l+2] = 1,1,1,1,1,1
            part_Fisher_bandpower_PP[~mask] = 0
        
        data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).ravel()
        SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
        #print SNR_PP
        SNR_PP_list.append(SNR_PP)

    return np.array(kklist), np.array(SNR_PP_list).ravel()



def CombineEstimator(RSDPower, rmin, rmax, kmin, kmax, n=False ):
    
    from numpy.linalg import inv
    from noshellavg import CombineDevXi, confidence_ellipse, FisherProjection_Fishergiven
    
    """ Two Step error plot and cofidenece ellipse " \
        should be seperated later cuz they use different number of bins """
    
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    rcut_min = get_closest_index_in_data( rmin, RSDPower.rmin )
    rcut_max = get_closest_index_in_data( rmax, RSDPower.rmax )
    
    print rcut_min, rcut_max
    
    matricesXi = [RSDPower.covariance00[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance02[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance22[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1],\
                  np.transpose(RSDPower.covariance04[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  np.transpose(RSDPower.covariance24[rcut_max:rcut_min+1,rcut_max:rcut_min+1]),\
                  RSDPower.covariance44[rcut_max:rcut_min+1,rcut_max:rcut_min+1]]
                  
    matricesPP_all = [RSDPower.covariance_PP00,RSDPower.covariance_PP02,\
                  RSDPower.covariance_PP04,RSDPower.covariance_PP02,\
                  RSDPower.covariance_PP22,RSDPower.covariance_PP24,\
                  RSDPower.covariance_PP04,RSDPower.covariance_PP24,\
                  RSDPower.covariance_PP44]
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]

    matricesPXi = [RSDPower.covariance_PXi00[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi02[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi04[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi20[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi22[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi24[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi40[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi42[kcut_min:kcut_max+1,rcut_max:rcut_min+1],\
                   RSDPower.covariance_PXi44[kcut_min:kcut_max+1,rcut_max:rcut_min+1]]

    
    
    
    l1 = RSDPower.kcenter_y.size
    l2 = rcut_min+1 - rcut_max
    l3 = kcut_max+1 - kcut_min

    
    # combining covariances
    C_matrix3PP_all = CombineCovariance3(l1, matricesPP_all)
    C_matrix3PP = CombineCovariance3(l3, matricesPP)
    C_matrix3PXi, C_matrix3XiP = CombineCrossCovariance3(l3, l2, matricesPXi, transpose = True)
    C_matrix3Xi = CombineCovariance3(l2, matricesXi)
    print C_matrix3Xi.shape
    
    C_matrix3PXi = np.zeros((C_matrix3PXi.shape))
    C_matrix3_tot = np.concatenate((np.concatenate((C_matrix3PP, C_matrix3PXi), axis=1), np.concatenate((C_matrix3PXi.T, C_matrix3Xi), axis=1)), axis = 0)
    
    
    """
    #fig = plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots()
    #ax = fig.add_axes()
    im = ax.imshow(CrossCoeff(C_matrix3_tot))
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
    ax1.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
    ax1.set_yticks([0., 0.2, 0.4, 0.6, 0.8])
    ax2 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
    ax2.set_xticklabels('')
    ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
    ax3.set_yticklabels('')
    ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
    ax4.set_xticklabels('')
    ax4.set_yticklabels('')
    """
    
    """
    import matplotlib.ticker as ticker
    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    klist = RSDPower.kcenter_y[kcut_min:kcut_max+1]
    extent = np.array([klist, klist, klist, RSDPower.rcenter, RSDPower.rcenter, RSDPower.rcenter]).ravel()
    im = ax.imshow(CrossCoeff(C_matrix3_tot))
    #ax.xaxis.set_major_locator(ticker.FixedLocator([50, 150, 250, 350, 450, 550]))
    #ax.xaxis.set_major_locator(ticker.FixedLocator(extent))
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.FixedLocator([50, 150, 250, 350, 450, 550]))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter([ 'CP 0', 'CP 2', 'CP 4', 'CXi 0', 'CXi 2', 'CXi 4']))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]))
    labels = ['1e-3', '0.03', '1e-3', '0.03', '1e-3','0.03', '1.0|0.1', '4.6', '0.1','4.6', '0.1', '4.6', '200' ]
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    ax.set_xlabel(' log k                                                   log r')
    ax.grid(b=True, which='major')
    fig.colorbar(im, ax=ax, label='correlation')
    """

    #inverse
    FisherP = inv(C_matrix3PP_all)
    #FisherXi = pinv(C_matrix3Xi)
    #Fisher3_tot = pinv(C_matrix3_tot)
    Fisher3_tot = blockwiseInversion( C_matrix3_tot, 3 * (kcut_max+1-kcut_min) )
    FisherXi = blockwiseInversion( C_matrix3Xi, rcut_min+1-rcut_max )

    
    # derivatives
    derivative_P0 = np.identity(RSDPower.kcenter_y.size)[:,kcut_min:kcut_max+1]
    Pzeros = np.zeros((derivative_P0.shape))
    
    
    derivative_P = np.concatenate((np.concatenate((derivative_P0, Pzeros, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, derivative_P0, Pzeros),axis=1 ),\
                                   np.concatenate((Pzeros, Pzeros, derivative_P0),axis=1 )), axis=0)
    Xizeros = np.zeros((RSDPower.dxip0[:,rcut_max:rcut_min+1].shape))
    derivative_correl_avg = np.concatenate(( np.concatenate((RSDPower.dxip0[:,rcut_max:rcut_min+1],Xizeros,Xizeros), axis=1),\
                                            np.concatenate((Xizeros,RSDPower.dxip2[:,rcut_max:rcut_min+1],Xizeros), axis=1),\
                                            np.concatenate((Xizeros,Xizeros,RSDPower.dxip4[:,rcut_max:rcut_min+1]), axis=1)),axis=0 )
    Derivatives = np.concatenate((derivative_P,derivative_correl_avg), axis=1)

    #projection
    FisherBand_P = FisherP.copy()
    FisherBand_Xi = FisherProjection_Fishergiven(derivative_correl_avg, FisherXi)
    FisherBand_tot = FisherProjection_Fishergiven(Derivatives, Fisher3_tot)



    # Power Spectrum Err -----------------------

    #all
    """
    errP0 = np.sqrt(C_matrix3PP_all.diagonal())
    errP2 = np.sqrt(C_matrix3PP_all.diagonal()[RSDPower.kcenter_y.size:2 * RSDPower.kcenter_y.size+1])
    errP4 = np.sqrt(C_matrix3PP_all.diagonal()[2 * RSDPower.kcenter_y.size:3 * RSDPower.kcenter_y.size+2])
    
    CovBandXi = pinv(FisherBand_Xi)
    errXi0 = np.sqrt(CovBandXi.diagonal())
    errXi2 = np.sqrt(CovBandXi.diagonal()[RSDPower.kcenter_y.size:2 * RSDPower.kcenter_y.size+1])
    errXi4 = np.sqrt(CovBandXi.diagonal()[2 * RSDPower.kcenter_y.size:3 * RSDPower.kcenter_y.size+2])

    CovBandtot = blockwiseInversion(FisherBand_tot, RSDPower.kcenter_y.size)
    errtot0 = np.sqrt(CovBandtot.diagonal())
    errtot2 = np.sqrt(CovBandtot.diagonal()[RSDPower.kcenter_y.size:2 * RSDPower.kcenter_y.size+1])
    errtot4 = np.sqrt(CovBandtot.diagonal()[2 * RSDPower.kcenter_y.size:3 * RSDPower.kcenter_y.size+2])
    
    #errtot = np.sqrt(C_matrix3_tot.diagonal()[0:RSDPower.kcenter_y.size])
    num = 5
    kk = np.array([RSDPower.kcenter_y[i] for i in range(0, RSDPower.kcenter_y.size, num)])
    P0 = np.array([RSDPower.multipole_bandpower0[i] for i in range(0, RSDPower.kcenter_y.size, num)])
    P2 = np.array([RSDPower.multipole_bandpower2[i] for i in range(0, RSDPower.kcenter_y.size, num)])
    P4 = np.array([RSDPower.multipole_bandpower4[i] for i in range(0, RSDPower.kcenter_y.size, num)])
    
    errP0 = [errP0[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errP2 = [errP2[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errP4 = [errP4[i] for i in range(0, RSDPower.kcenter_y.size, num)]

    errXi0 = [errXi0[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errXi2 = [errXi2[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errXi4 = [errXi4[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    
    errtot0 = [errtot0[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errtot2 = [errtot2[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    errtot4 = [errtot4[i] for i in range(0, RSDPower.kcenter_y.size, num)]
    
    fig, ((ax, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(30, 20))
    
    ax.errorbar(kk, P0, yerr = errP0, label='P')
    ax.errorbar(kk*1.05, P0, yerr = errXi0, fmt=None , label='Xi')
    ax.errorbar(kk*0.95, P0, yerr = errtot0, fmt=None, label='combine')
    ax.set_title('P multipole 0')
    ax.set_xlabel('k (h/Mpc)')
    ax.set_ylabel('P0(k)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-3, 2)
    ax.legend(loc='best')
    
    ax2.errorbar(kk, P2, yerr = errP2)
    ax2.errorbar(kk*1.05, P2, yerr = errXi2, fmt=None)
    ax2.errorbar(kk*0.95, P2, yerr = errtot2, fmt=None)
    ax2.set_title('P multipole 2')
    ax2.set_xlabel('k (h/Mpc)')
    ax2.set_ylabel('P2(k)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e-3, 2)
    
    ax3.errorbar(kk, P4, yerr = errP4)
    ax3.errorbar(kk*1.05, P4, yerr = errXi4, fmt=None)
    ax3.errorbar(kk*0.95, P4, yerr = errtot4, fmt=None)
    ax3.set_title('P multipole 4')
    ax3.set_xlabel('k (h/Mpc)')
    ax3.set_ylabel('P4(k)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(1e-3, 2)
    
    ax4.errorbar(kk, P0, yerr = errP0)
    ax4.errorbar(kk*1.05, P0, yerr = errXi0, fmt=None , label='Xi')
    ax4.errorbar(kk*0.95, P0, yerr = errtot0, fmt=None, label='combine')
    ax4.set_title('P multipole 0')
    ax4.set_xlabel('k (h/Mpc)')
    ax4.set_ylabel('P0(k)')
    ax4.set_xscale('linear')
    ax4.set_yscale('log')
    
    ax5.errorbar(kk, P2, yerr = errP2)
    ax5.errorbar(kk*1.05, P2, yerr = errXi2, fmt=None)
    ax5.errorbar(kk*0.95, P2, yerr = errtot2, fmt=None)
    ax5.set_title('P multipole 2')
    ax5.set_xlabel('k (h/Mpc)')
    ax5.set_ylabel('P2(k)')
    ax5.set_xscale('log')
    ax5.set_ylim(-10000, 50000)
    #ax5.set_yscale('log')
    ax5.set_xlim(1e-3, 2)
    
    ax6.errorbar(kk, P4, yerr = errP4)
    ax6.errorbar(kk*1.05, P4, yerr = errXi4, fmt=None)
    ax6.errorbar(kk*0.95, P4, yerr = errtot4, fmt=None)
    ax6.set_title('P multipole 4')
    ax6.set_xlabel('k (h/Mpc)')
    ax6.set_ylabel('P4(k)')
    ax6.set_xscale('log')
    ax6.set_ylim(-500, 4000)
    ax6.set_xlim(1e-3, 2)
    #ax6.set_yscale('log')

    #fig.savefig('Pmonopole')
    """
    
    """
    fig2, (ax, ax2) = plt.subplots(1,2)
    im = ax.imshow(CrossCoeff(FisherBand_Xi))
    im2 = ax2.imshow(CrossCoeff(FisherBand_tot))
    fig2.colorbar(im, ax=ax)
    fig2.colorbar(im2, ax=ax2)
    """
    
    # SNR ---------------------------
    
    """
    PP = RSDPower.multipole_bandpower0
    FXi00 = np.dot(np.dot(RSDPower.dxip0, inv(RSDPower.covariance00)), RSDPower.dxip0.T)
    SNRlist00 = []
    SNRlist00.append(np.dot( np.dot(PP, FXi00), PP.T ))
    
    Ftot, P = reordering( RSDPower, FisherBand_tot )
    FP, _ = reordering( RSDPower, FisherBand_P )
    FXi, _ = reordering( RSDPower, FisherBand_Xi )
    SNRlist_tot = []
    SNRlist_P = []
    SNRlist_Xi = []
    SNRlist_tot.append(np.dot( np.dot(P, Ftot), P.T ))
    SNRlist_P.append(np.dot( np.dot(P, FP), P.T ))
    SNRlist_Xi.append(np.dot( np.dot(P, FXi), P.T ))

    for j in range(1, RSDPower.kcenter_y.size):
        P = P[:-3]
        PP = PP[:-1]
        FXi00 = blockwise(FXi00)
        SNRlist00.append( np.dot( np.dot( PP, FXi00), PP.T))
        for i in range(0,3):
            
            Ftot = blockwise( Ftot )
            FP = blockwise( FP )
            FXi = blockwise( FXi )
            #print FXi, P
        SNRlist_tot.append(np.dot( np.dot(P, Ftot), P.T ))
        SNRlist_P.append(np.dot( np.dot(P, FP), P.T ))
        SNRlist_Xi.append(np.dot( np.dot(P, FXi), P.T ))

    SNRlist00 = np.array(SNRlist00[::-1]).ravel()
    SNRlist_tot = np.array(SNRlist_tot[::-1]).ravel()
    SNRlist_P = np.array(SNRlist_P[::-1]).ravel()
    SNRlist_Xi = np.array(SNRlist_Xi[::-1]).ravel()




    fig3, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2,2, figsize = (14,14))
    #fig3, ax3 = plt.subplots()
    fig3.suptitle( 'SNR RSD scale ( r24-152, k0.01-0.2 )')
    #fig3.suptitle( 'SNR BAO scale ( r29-200, k0.02-0.3 )')
    fig3.suptitle( 'SNR ( r0.1-200, k0.001-1 )')
    ax3.plot(RSDPower.kcenter_y, SNRlist_P, color = 'grey', linestyle='--')
    ax3.plot(RSDPower.kcenter_y[kcut_min:kcut_max+1], SNRlist_P[kcut_min:kcut_max+1], 'k-', label = 'SNR P')
    ax3.scatter(RSDPower.kcenter_y, SNRlist_Xi, color = 'red', s=1, label = 'SNR Xi')
    ax3.scatter(RSDPower.kcenter_y, SNRlist_tot, color='blue', s=1,label = 'SNR tot')
    ax3.set_ylim(0, 600000)
    #ax3.set_xlim(0,1)
    ax3.set_xscale('linear')
    #ax3.set_yscale('log')
    
    ax3.set_title('SNR')
    ax3.set_xlabel('k')
    ax3.set_ylabel('(S/N)^2')
    ax3.legend(loc = 'best')

    ax4.scatter(RSDPower.kcenter_y, SNRlist00, color = 'green', s = 1, label = 'SNR Xi0')
    #ax4.plot(RSDPower.kcenter_y[kcut_min:kcut_max+1], SNRlist_P[kcut_min:kcut_max+1], 'k-', label = 'SNR P')
    ax4.scatter(RSDPower.kcenter_y, SNRlist_Xi, color = 'red', s = 1, label = 'SNR Xi')
    #ax4.scatter(RSDPower.kcenter_y, SNRlist_tot, color='blue', s=5,label = 'SNR tot')
    ax4.set_ylim(0, 8000)
    #ax4.set_xlim(0,1)
    ax4.set_xscale('linear')
    #ax4.set_yscale('log')
    
    ax4.set_title('SNR')
    ax4.set_xlabel('k')
    ax4.set_ylabel('(S/N)^2')
    #ax4.legend(loc = 'best')
    
    ax5.plot(RSDPower.kcenter_y, SNRlist_P, color = 'grey', linestyle='--')
    ax5.plot(RSDPower.kcenter_y[kcut_min:kcut_max+1], SNRlist_P[kcut_min:kcut_max+1], 'k-', label = 'SNR P')
    ax5.scatter(RSDPower.kcenter_y, SNRlist_Xi, color = 'red', s = 1, label = 'SNR Xi')
    ax5.scatter(RSDPower.kcenter_y, SNRlist_tot, color='blue', s=1,label = 'SNR tot')
    ax5.set_ylim(1e-10, 600000)
    ax5.set_xlim(0.001,1)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_title('SNR')
    ax5.set_xlabel('log scale k')
    ax5.set_ylabel('(S/N)^2')
    #ax5.legend(loc = 'best')
    """
    
    """
    ax6.plot(RSDPower.kcenter_y, SNRlist_P, 'k--')
    ax6.plot(RSDPower.kcenter_y[kcut_min:kcut_max+1], SNRlist_P[kcut_min:kcut_max+1], 'k-', label = 'SNR P')
    ax6.scatter(RSDPower.kcenter_y, SNRlist_Xi, color = 'red', s = 5, label = 'SNR Xi')
    ax6.scatter(RSDPower.kcenter_y, SNRlist_tot, color='blue', s=5,label = 'SNR tot')
    ax6.set_ylim(1e-2, 7000)
    ax6.set_xlim(0.001,1)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_title('SNR')
    ax6.set_xlabel('log scale k')
    ax6.set_ylabel('(S/N)^2')
    #ax6.legend(loc = 'best')
    """


    # SNR function of rmin --------------------------------
    CXi = np.vstack((np.hstack((RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04)),
                     np.hstack(( RSDPower.covariance02.T, RSDPower.covariance22,RSDPower.covariance24)),
                     np.hstack(( RSDPower.covariance04.T,RSDPower.covariance24.T,RSDPower.covariance44))))
    
    CXi, _ = reordering(RSDPower, CXi, cut = RSDPower.rcenter.size )
    CP = CombineCovariance3(RSDPower.kcenter_y.size, matricesPP_all)
    CP, reorderedP = reordering(RSDPower, CP, cut = RSDPower.kcenter_y.size)
    
    Xizeros = np.zeros((RSDPower.dxip0.shape))
    dxip0 = reorderingVector([RSDPower.dxip0, Xizeros, Xizeros])
    dxip2 = reorderingVector([Xizeros, RSDPower.dxip2, Xizeros])
    dxip4 = reorderingVector([Xizeros,Xizeros, RSDPower.dxip4])
    dxip = np.vstack(( dxip0, dxip2, dxip4 ))
    P = np.array([RSDPower.multipole_bandpower0, RSDPower.multipole_bandpower2, RSDPower.multipole_bandpower4]).ravel()
    
    rrlist= []
    SNR = []
    for l in range(1, RSDPower.rcenter.size):
        C = CXi[0:3 * l, 0:3 * l]
        FXi = inv(C)
        dx = dxip[:, 0:3 * l]
        Fisher = np.dot(np.dot(dx, FXi), dx.T)
        snr = np.dot( np.dot( P, Fisher ), P.T )
        SNR.append(snr)
        rrlist.append( RSDPower.rcenter[l] )
    
    rrlist2 = []
    SNR2 = []
    for l in range(1, RSDPower.kcenter_y.size):
        C = CP[0:3 * l, 0:3 * l]
        FP = inv(C)
        snr = np.dot( np.dot( reorderedP[0:3*l], FP ), reorderedP[0:3*l].T )
        SNR2.append(snr)
        rrlist2.append(np.pi / RSDPower.kcenter_y[l])
    
 


    # SNR function of r min multipoles

    
    rrlist_m= [[], [], []]
    SNR_m = [[], [], []]
    
    dx = [RSDPower.dxip0, RSDPower.dxip2, RSDPower.dxip4]
    C = [RSDPower.covariance00, RSDPower.covariance22, RSDPower.covariance44 ] 
    P = [RSDPower.multipole_bandpower0,RSDPower.multipole_bandpower2,RSDPower.multipole_bandpower4 ]

    for i in range(3):
        for l in range(1, RSDPower.rcenter.size):
            Fisher = np.dot(np.dot(dx[i][:,0:l], inv(C[i][0:l, 0:l])), np.array(dx[i][:,0:l]).T)
            snr = np.dot( np.dot( P[i], Fisher ), P[i].T )
            SNR_m[i].append(snr)
            rrlist_m[i].append( RSDPower.rcenter[l] )


    C = [RSDPower.covariance_PP00, RSDPower.covariance_PP22, RSDPower.covariance_PP44 ] 
    rrlist2_m = [[], [], []]
    SNRP_m = [[], [], []]
    for i in range(3):
        for l in range(1, RSDPower.kcenter_y.size):
            FP = inv(C[i][0:l, 0:l])
            snr = np.dot( np.dot( P[i][0:l], FP ), P[i][0:l].T )
            SNRP_m[i].append(snr)
            rrlist2_m[i].append(np.pi / RSDPower.kcenter_y[l])
    

    # save
    DAT1 = np.column_stack((rrlist, SNR, rrlist_m[0], SNR_m[0], rrlist_m[1],SNR_m[1], rrlist_m[2], SNR_m[2]))
    DAT2 = np.column_stack(( rrlist2, SNR2, rrlist2_m[0], SNRP_m[0], rrlist2_m[1],SNRP_m[1], rrlist2_m[2], SNRP_m[2] ))
    np.savetxt('snr_figure/snr_rcut.txt', DAT1, delimiter = ' '  )
    np.savetxt('snr_figure/snr_rcutP.txt', DAT2, delimiter = ' '  )

    # plott
    fig, ax = plt.subplots(1,1, figsize = (7, 7))

    ax.plot(rrlist, 1./np.sqrt(SNR), 'b-',label = 'SNR Xi' )
    ax.plot(rrlist2, 1./np.sqrt(SNR2), 'b--',label = 'SNR P' )

    labels1 = ['Xi0', 'Xi2', 'Xi4']
    labels2 = ['P0', 'P2', 'P4']
    c = ['red', 'cyan', 'green']
    for i in range(len(labels1)):
        ax.plot(rrlist_m[i], 1./np.sqrt(SNR_m[i]), '-', color =c[i]  , label = labels1[i])
        ax.plot(rrlist2_m[i], 1./np.sqrt(SNRP_m[i]), '--', color =c[i], label = labels2[i])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xlim(1e-1, 500)
    ax.set_xlabel('r_min, Pi/k_max')
    #ax.set_ylabel('fractional error P={:>0.2f}h/Mpc'.format(P[RSDPower.kcenter_y.size/2]))
    ax.legend(loc = 'best')
    
    
    

    stop
    return rrlist, SNR, rrlist2, SNRP








    # Reid rcut test ------------------
    
    from noshellavg import CombineDevXi

    
    C_matrix3Xi, _ = reordering(RSDPower, C_matrix3Xi, cut = RSDPower.rcenter[rcut_max:rcut_min+1].size )

    Xizeros = np.zeros((RSDPower.dxip0[:, rcut_max:rcut_min+1].shape))
    dxip0 = reorderingVector([RSDPower.dxip0[:, rcut_max:rcut_min+1], Xizeros, Xizeros])
    dxip2 = reorderingVector([Xizeros, RSDPower.dxip2[:, rcut_max:rcut_min+1], Xizeros])
    dxip4 = reorderingVector([Xizeros,Xizeros, RSDPower.dxip4[:, rcut_max:rcut_min+1]])
    dxip = np.vstack(( dxip0, dxip2, dxip4 ))

    dPb = np.hstack(([RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4]))
    dPf = np.hstack(([RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4]))
    dPs = np.hstack(([RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]))
    dPbfs = np.vstack(( dPb, dPf, dPs ))
    
    Fisherb = []
    Fisherf = []
    Fisherb_det = []
    Fisherf_det = []
    rrlist = []
    
    FisherXi = inv(C_matrix3Xi)
    
    
    Fisher = np.dot(np.dot(dxip, FisherXi), dxip.T)
    FReid = np.dot( np.dot( dPbfs, Fisher), dPbfs.T )
    Cov = inv(FReid)
    Cov_det = inv(FReid[0:2, 0:2])
    Fisherb.append(np.sqrt(Cov[0,0]))
    Fisherf.append(np.sqrt(Cov[1,1]))
    Fisherb_det.append(np.sqrt(Cov_det[0,0]))
    Fisherf_det.append(np.sqrt(Cov_det[1,1]))
    reverser = RSDPower.rcenter[rcut_max:rcut_min+1][::-1]
    rrlist.append( reverser[0])




    for l in range(1, rcut_min+1-rcut_max, 2):
        C_matrix3 = C_matrix3Xi[0:-3 * l, 0:-3 * l]
        FisherXi = inv(C_matrix3)
        dx = dxip[:, 0:-3 * l]
        Fisher = np.dot(np.dot(dx, FisherXi), dx.T)
        FReid = np.dot( np.dot( dPbfs, Fisher), dPbfs.T )
        Cov = inv(FReid)
        Cov_det = inv(FReid[0:2, 0:2])
        Fisherb.append(np.sqrt(Cov[0,0]))
        Fisherf.append(np.sqrt(Cov[1,1]))
        Fisherb_det.append(np.sqrt(Cov_det[0,0]))
        Fisherf_det.append(np.sqrt(Cov_det[1,1]))
        rrlist.append( reverser[l])

    rrlist = np.array(rrlist).ravel()
    Fisherb = np.array(Fisherb).ravel()/RSDPower.b
    Fisherf = np.array(Fisherf).ravel()/RSDPower.f
    Fisherb_det = np.array(Fisherb_det).ravel()/RSDPower.b
    Fisherf_det = np.array(Fisherf_det).ravel()/RSDPower.f


    PFisherb = []
    PFisherf = []
    PFisherb_det = []
    PFisherf_det = []
    Prrlist = []

    matrices2P = [RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4, RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4, RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]

    for l in range(1, RSDPower.kcenter_y.size, 2):
        C_matrix3PP_all = CombineCovariance3(l, matricesPP_all)
        XP, XP2 = CombineDevXi(l, matrices2P)
        FisherXi = np.dot( np.dot( XP, inv(C_matrix3PP_all)), XP.T)
        Cov = inv(FisherXi)
        Cov_det = inv(FisherXi[0:2, 0:2])
        PFisherb.append(np.sqrt(Cov[0,0]))
        PFisherf.append(np.sqrt(Cov[1,1]))
        PFisherb_det.append(np.sqrt(Cov_det[0,0]))
        PFisherf_det.append(np.sqrt(Cov_det[1,1]))
        Prrlist.append(1.15 * np.pi/ RSDPower.kcenter_y[l])

    Prrlist = np.array(Prrlist).ravel()
    PFisherb = np.array(PFisherb).ravel()/RSDPower.b
    PFisherf = np.array(PFisherf).ravel()/RSDPower.f
    PFisherb_det = np.array(PFisherb_det).ravel()/RSDPower.b
    PFisherf_det = np.array(PFisherf_det).ravel()/RSDPower.f
    
    
    fig, (ax, ax2) = plt.subplots(2,1)
    
    ax.semilogy( rrlist, Fisherb , 'r--', label = 's marginalized')
    ax.semilogy( rrlist, Fisherb_det , 'r-', label = 's determined')
    ax.semilogy( Prrlist, PFisherb_det , 'k-.', label = 'P')

    ax2.plot( rrlist, Fisherf, 'b--', label = 's marginalized')
    ax2.plot( rrlist, Fisherf_det, 'b-', label = 's determined')
    ax2.plot( Prrlist, PFisherf_det , 'k-.', label = 'P')

    Xrrlist, XFisherb, XFisherf, XFisherb_det, XFisherf_det = ReidResult(RSDPower,rmin, rmax, kmin, kmax)

    ax.semilogy( Xrrlist, XFisherb , 'g--', label = 'Xi s marginalized')
    ax.semilogy( Xrrlist, XFisherb_det , 'g-', label = 'Xi s determined')

    ax2.plot( Xrrlist, XFisherf, 'k--', label = 'Xi s marginalized')
    ax2.plot( Xrrlist, XFisherf_det, 'k-', label = 'Xi s determined')

    ax.set_xlim(0, 60)
    ax.set_ylim(0.0005, 0.05)
    ax.set_ylabel('b')
    ax.set_xlabel('r (Mpc/h)')
    ax2.set_xlim(0,60)
    ax2.set_ylim(0.00, 0.07)
    ax2.set_ylabel('f')
    ax2.set_xlabel('r (Mpc/h)')

    ax.legend(loc = 'best')
    ax2.legend(loc='best')
    ax.set_title(' from F_bandpower ')


    # Ellipse--------------

    matrices2P_cut = [RSDPower.dPb0[kcut_min:kcut_max+1], RSDPower.dPb2[kcut_min:kcut_max+1],\
                      RSDPower.dPb4[kcut_min:kcut_max+1], RSDPower.dPf0[kcut_min:kcut_max+1],\
                      RSDPower.dPf2[kcut_min:kcut_max+1], RSDPower.dPf4[kcut_min:kcut_max+1],\
                      RSDPower.dPs0[kcut_min:kcut_max+1], RSDPower.dPs2[kcut_min:kcut_max+1],\
                      RSDPower.dPs4[kcut_min:kcut_max+1]]
    
    matrices2P = [RSDPower.dPb0, RSDPower.dPb2, RSDPower.dPb4, RSDPower.dPf0, RSDPower.dPf2, RSDPower.dPf4, RSDPower.dPs0, RSDPower.dPs2, RSDPower.dPs4]
    
    XP, XP2 = CombineDevXi(l1, matrices2P)
    XP_cut, XP2_cut = CombineDevXi(l3, matrices2P_cut)

    n = True
    if n == True:
        dPN0 = np.ones(RSDPower.kcenter_y.size)
        dPN1 = np.zeros(RSDPower.kcenter_y.size)
        dPN2 = dPN1.copy()
        XP = np.vstack((XP,np.array([dPN0, dPN1, dPN2]).ravel()))
        XP_cut = np.vstack((XP_cut, np.array([dPN0[kcut_min:kcut_max+1], dPN1[kcut_min:kcut_max+1], dPN2[kcut_min:kcut_max+1]]).ravel()))
        #matrices2P = matrices2P + [dPN0, dPN1, dPN2]
        #matrices2P_cut = matrices2P_cut + [dPN0[kcut_min:kcut_max+1], dPN1[kcut_min:kcut_max+1], dPN2[kcut_min:kcut_max+1]]
    
    
    FisherPP = np.dot( np.dot( XP_cut, inv(C_matrix3PP)), XP_cut.T)
    FisherXi = np.dot( np.dot( XP, FisherBand_Xi), XP.T)
    Fishertot = np.dot( np.dot( XP, FisherBand_tot), XP.T)
    
    Cov_PP = inv(FisherPP)[0:2,0:2]
    Cov_Xi = inv(FisherXi)[0:2,0:2]
    Cov_tot = inv(Fishertot)[0:2,0:2]

    elllist = confidence_ellipse(RSDPower.b, RSDPower.f, Cov_PP, Cov_Xi, Cov_tot)

    fig, ax = plt.subplots(figsize=(10,7))
    for e in elllist:
        ax.add_artist(e)
        #e.set_alpha(0.2)
        e.set_clip_box(ax.bbox)

    xmin = RSDPower.b*0.97
    xmax = RSDPower.b*1.03
    ymin = RSDPower.f*0.92
    ymax = RSDPower.f*1.08
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('b')
    ax.set_ylabel('f')
    labellist = ['P', 'Xi', 'tot']
    ax.legend(elllist, labellist, loc=4, prop={'size':10})
    #ax.set_title( 'error ellipse of b and f, s marginalized, RSD scale ( r24-152, k0.01-0.2 )' )
    #ax.set_title( 'error ellipse of b and f, s marginalized, BAO scale ( r29-200, k0.02-0.3 )' )
    ax.set_title( 'error ellipse of b and f, s marginalized, ( r0.1-200, k0.001-1 )' )
    fig.savefig('figure/ellipse.png')
    plt.close(fig)


def Cumulative_SNR_loop(RSDPower, kmin, kmax, l):
    """
    Calculate cumulative Signal to Noise up to kmax
    
    Parameter
    ---------
    RSDPower: class name
    l: slicing index. determin kmax=k[l]
    
    """
    
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax )

    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    matricesPP = [RSDPower.covariance_PP00[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP02[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP22[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP04[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP24[kcut_min:kcut_max+1,kcut_min:kcut_max+1],
                  RSDPower.covariance_PP44[kcut_min:kcut_max+1,kcut_min:kcut_max+1]]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    
    matrices2Xi = [RSDPower.dxip0[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , RSDPower.dxip2[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , Xizeros[kcut_min:kcut_max+1,:]
                   , RSDPower.dxip4[kcut_min:kcut_max+1,:]]
    
    multipole_P0, multipole_P2, multipole_P4 = RSDPower.multipole_bandpower0[kcut_min:kcut_max+1], RSDPower.multipole_bandpower2[kcut_min:kcut_max+1], RSDPower.multipole_bandpower4[kcut_min:kcut_max+1]
    # F_bandpower from P
    part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
    
    # GET full C_Xi
    l_r = len(RSDPower.rcenter)
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)  # Here, cut out small and large k
    
    # F_bandpower from Xi
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    Cov_bandpower_Xi = inv( Fisher_bandpower_Xi ) #, rcond = 1e-15 )
    
    #a = Fisher_bandpower_Xi
    #B = Cov_bandpower_Xi
    #check = np.allclose(a, np.dot(a, np.dot(B, a)))         # for pinv
    #check = np.allclose(np.dot(a, B), np.eye(a.shape[0]))   # for inv
    #if check == False : print 'inverting failed'
    
    cut = kcut_max + 1- kcut_min #len(RSDPower.kcenter)
    part00 = Cov_bandpower_Xi[0:cut, 0:cut]
    part02 = Cov_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Cov_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Cov_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Cov_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Cov_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]
    
    part_list = [ part00, part02, part04, np.transpose(part02), part22, part24, np.transpose(part04), np.transpose(part24), part44]
    Cov_bandpower_Xi_combine = CombineCovariance3(l, part_list)
    part_Fisher_bandpower_Xi = inv( Cov_bandpower_Xi_combine ) #, rcond = 1e-15)
    
    #a = Cov_bandpower_Xi_combine
    #B = part_Fisher_bandpower_Xi
    #check = np.allclose(np.dot(a, B), np.eye(a.shape[0]))
    #check = np.allclose(a, np.dot(a, np.dot(B, a)))
    #print check, '3rd'
    
    
    
    
    
    
    data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).ravel()
    
    data_Vec0 = RSDPower.multipole_bandpower0[0:l+1]
    Fisher_bandpower_PP00 = inv(RSDPower.covariance_PP00[0:l+1,0:l+1])
    
    SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
    SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), np.transpose(data_Vec))
    
    
    return RSDPower.kcenter[kcut_min:kcut_max+1][l], SNR_PP, SNR_Xi



def SNR_multiprocessing(RSDPower, kmin, kmax):
    """
    Do multiprocessing for the function Cumulative_SNR_loop()
    
    Parameter
    ---------
    RSDPower: class name
    kcut_max: index of kmax
    
    """
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax )
    
    num_process = 12
    
    #print 'multi_processing for k loop : ', num_process, ' workers'
    numberlist_k = np.arange(1, kcut_max-1,100)
    numberlist_k_split = np.array_split(numberlist_k, num_process)
    
    vec_Cumulative_SNR_loop = np.vectorize(Cumulative_SNR_loop)
    
    
    def cumulative_SNR_all(q, order, input):
    
        kklist, SNR_PP_list, SNR_Xi_list = vec_Cumulative_SNR_loop(RSDPower, kmin, kmax, input)
        DAT = np.array(np.concatenate(( kklist, SNR_PP_list, SNR_Xi_list ), axis = 0)).reshape(3,len(input))
        q.put(( order, DAT ))
        sys.stdout.write('.')

    loop_queue = Queue()

    loop_processes = [Process(target=cumulative_SNR_all, args=(loop_queue, z[0], z[1])) for z in zip(range(num_process+1), numberlist_k_split)]
    
    for p in loop_processes:
        p.start()

    loop_result = [loop_queue.get() for p in loop_processes]
    loop_result.sort()
    loop_result_list = [ loop[1] for loop in loop_result ]
    loops = loop_result_list[0]
    for i in range(1, num_process):
        loops = np.concatenate((loops, loop_result_list[i]), axis = 1 )

    kklist = loops[0]
    SNR_PP_list = loops[1]
    SNR_Xi_list = loops[2]
    
    #print "done \nSNR final 10 values :", SNR_Xi_list[-11:-1]
    print 'done'
    return kklist, SNR_PP_list, SNR_Xi_list





def main_SNR(rmin = None, rmax = None):
    #from noshellavg import *



    # initial setting ------------------------------
    
    #  (parameter description in class code)
    # Fourier K 0.001~10
    KMIN = 0.001
    KMAX = 2. #361.32 #502.32
    RMIN = 0.1
    RMAX = 400
    
    # the number of k sample point should be 2^n+1 (b/c romb integration)
    rN = 300
    kN = 1
    kN_x = 2**13 + 1
    kN_y = 71
    #subN = 2**5 + 1
    # RSD class



    def ComputeRSDClass():
        #RSDPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN, N_x, logscale = False)
        RSDPower = NoShell_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, kN_x, kN_y, logscale = True)
        #RSDPower = Even_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN, N_x, logscale = False)
        #RSDPower.compile_fortran_modules() # compile only one time
        # make bandpower and cov matrices-----------------------
        # (function description in class code)
        file = 'matterpower_z_0.55.dat'  # from camb (z=0.55)
        RSDPower.MatterPower(file = file)
    	#RSDPower.Shell_avg_band()
    
        print '\nStarting multiprocessing (about 60 sec for test code)'
    	# power spectrum multipoles l = 0,2,4
        RSDPower.multipole_P_band_all()
    
    	# derivative dXi/dp
        RSDPower.derivative_Xi_band_all()

    	# derivative dXidb, s, f
        RSDPower.derivative_bfs_all()
        RSDPower.derivative_P_bfs_all()
    
    	# P covariance matrix ( nine submatrices C_ll' )
        RSDPower.RSDband_covariance_PP_all()
    
    	# Xi covariance matrix ( nine submatrices C_ll' )
        RSDPower.covariance_Xi_all()
        RSDPower.covariance_PXi_All()
        return RSDPower
    
    RSDPower = ComputeRSDClass()

    # cutting -------------------------------------------------
    kmin = KMIN
    kmax = KMAX
    rmin = RMIN
    rmax = RMAX
    rcut_max = get_closest_index_in_data( rmax, RSDPower.rmax )
    rcut_min = get_closest_index_in_data( rmin, RSDPower.rmin )
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin_y )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax_y )
    
    print "kcut_min :", RSDPower.kmin_y[kcut_min], "  kcut_max :", RSDPower.kmax_y[kcut_max]
    print "rcut_min :", RSDPower.rmin[rcut_min], "  rcut_max :", RSDPower.rmax[rcut_max]
    print 'kN', kN_y #kcut_max + 1 - kcut_min
    print 'rN', rcut_min + 1 - rcut_max
    #-------------------------------------------------------

    #kklist, BinSNRP = BinAvgSNR_P(RSDPower, kmin, kmax)
    
    fig, ax = plt.subplots()
    kkk = np.logspace(np.log10(kmin), np.log10(kmax), 20)
    for i in range(kkk.size -1):
        label = '[{:>0.2f}, {:>0.2f}]'.format(kkk[i], kkk[i+1])
        kklist, SNRPP = convergence_P(RSDPower, kkk[i], kkk[i+1])
        #kklist, SNRPP = BinAvgSNR_P(RSDPower, kkk[i], kkk[i+1])
        kklist2, SNRXI = convergence_Xi(RSDPower, rmin, rmax, kkk[i], kkk[i+1])
        ax.plot(kklist, SNRPP/kklist, linestyle = '--', label = label)
        ax.plot(kklist2, SNRXI/kklist2, linestyle = '-', color = ax.lines[-1].get_color())
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('k')
    ax.set_ylim(1e-12, 1e+10)
    ax.set_ylabel('Delta SNR/k')
    ax.set_title('SNR P (dashed) Xi(solid)')
    ax.legend(loc='best')
    fig.savefig('figure/DeltaSNR2.png')
    plt.close(fig)



    #rrlist, SNR, rrlist2, SNR2 = CombineEstimator(RSDPower, rmin, rmax, kmin, kmax)

    def computeXiSNR():
        rmins = [ 0.1, 10, 20]
        rmaxs= [200, 400 ]
        
        labels = []
        data_stacked = []
        for rmin in rmins:
            for rmax in rmaxs:
                print rmin, rmax
                kk, SNR, _ = convergence_Xi(RSDPower, rmin, rmax, kmin, kmax)
                data_stacked.append(kk)
                data_stacked.append(SNR)
                labels.append('rmin '+str(rmin) +', rmax '+ str(rmax))
        
        data_stacked = np.vstack(data_stacked).T
        return data_stacked, labels


    #kklistP, SNRPP = convergence_P(RSDPower, kmin, kmax)

    def plotting( Xidata, labels, suffix = None ):
        path = 'snr_figure/'
        DAT = np.loadtxt(path+Xidata)
        DAT2 = np.loadtxt(path+'snrP.txt')

        fig, (ax, ax2) = plt.subplots(1,2,figsize = (20, 10))

        for i in range(len(labels)):
            ax.plot(DAT[:, 2*i], DAT[:,2*i+1], '.', label = labels[i])
            ax2.plot(DAT[:, 2*i], DAT[:,2*i+1], '.', label = labels[i])

        ax.plot(DAT2[:,0], DAT2[:,1], 'k--', label = 'SNR P')
        ax2.plot(DAT2[:,0], DAT2[:,1], 'k--', label = 'SNR P')
    
        ax.set_ylim(1e-23, 1e7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #text_label = 'kN {} \ndlnk {:>0.4f} \ndlnr {:>0.4f}'.format(RSDPower.N_y, RSDPower.dlnk_y, RSDPower.dlnr)
        #ax.text(0.05, 0.95, text_label, ha='left', va='top', transform=ax.transAxes, fontsize = 10)
        ax.legend(loc='best', prop={'size':10})
        
        fig.savefig(path+Xidata+suffix)
 

    kN_y = 50
    rN = 400
    RSDPower = ComputeRSDClass()

    DAT, labels = computeXiSNR()
    np.savetxt('snr_figure/snrXi_'+str(kN_y)+'_'+str(rN)+'.txt', DAT, delimiter = ' '  )



    labels = []
    Xidata = []

    labels = ['rmin 0.1, rmax 200', 'rmin 0.1, rmax 400', 'rmin 10, rmax 200', 'rmin 10, rmax 400', 'rmin 20, rmax 200', 'rmin 20, rmax 400']

    print "calling data...."

    path = 'snr_figure/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and i.endswith('.txt') and 'Xi' in i:
            print i
            plotting(i, labels)
        



    stop

    #ReidResult(RSDPower,rmin, rmax, kmin, kmax)
    
    # combined total covariance
    # CPxi
    
    
    #



    stop
    
    return kklist, SNRPP

    
    # cumulative SNR and plotting ----------------------------

    #kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(RSDPower, kcut_max)
    #kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(LogPower, kcut_max_l)

    
    
    """
    
    
    kklist, SNRXI = main( rmin = 0.1, rmax = 200 )
    kklist0, SNRXI0 = main( rmin = 0.1, rmax = 400 )
    kklist1, SNRXI1 = main( rmin = 10, rmax = 200 )
    kklist2, SNRXI2 = main( rmin = 20, rmax = 200 )
    kklist3, SNRXI3 = main( rmin = 10, rmax = 400 )
    kklist4, SNRXI4 = main( rmin = 20, rmax = 400 )


    fig, (ax, ax2 ) = plt.subplots(1,2, figsize = (14,7))
    ax.plot( kklist, SNRXI, 'k.', label='0-200' )
    ax.plot( kklist0, SNRXI0, 'c.', label='0-400' )
    ax.plot( kklist1, SNRXI1, 'r.', label='10-200' )
    ax.plot( kklist2, SNRXI2, 'b.', label='20-200' )
    ax.plot( kklist3, SNRXI3, 'g.', label='10-400' )
    ax.plot( kklist4, SNRXI4, 'm.', label='20-400' )
    ax.set_title('k=0.01-2')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-22, 1e4)
    ax.legend(loc='best')
    fig.savefig('test1')





    ax2.plot( kklist13, SNRXI13, 'k.', label='0-200, dlnr=, Nk201' )
    ax2.plot( kklist14, SNRXI14, 'c.', label='0-400, dlnr=, Nk201' )
    ax2.plot( kklist9, SNRXI9, 'r.', label='10-200, dlnr=0.009, Nk201' )
    ax2.plot( kklist10, SNRXI10, 'b.', label='20-200, dlnr=0.008, Nk201' )
    ax2.plot( kklist11, SNRXI11, 'g.', label='10-400, dlnr=0.012, Nk201' )
    ax2.plot( kklist12, SNRXI12, 'm.', label='20-400, dlnr=0.009, Nk201' )
    ax2.set_title('k=0.01-2,  equal Nr = 300,  dlnk = 0.038')
    #ax2.set_ylim(-20000, 120000)
    ax2.legend(loc='best')

    """




if __name__=='__main__':
    main()
