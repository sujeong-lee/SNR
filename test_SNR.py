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


import numpy as np
from numpy import zeros, sqrt, pi
from numpy.linalg import pinv, inv
from numpy import vectorize
import time, datetime
from multiprocessing import Process, Queue
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from error_analysis_class import *


def DataSave(RSDPower, suffix = None):


    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    matricesPP = [RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04,RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24,RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    matrices2Xi = [RSDPower.dxip0, Xizeros,Xizeros,Xizeros,RSDPower.dxip2,Xizeros,Xizeros,Xizeros,RSDPower.dxip4]



    l = len(RSDPower.kcenter)
    karray = np.array([RSDPower.kcenter,RSDPower.kcenter,RSDPower.kcenter]).ravel()
    
    #print l, len(karray),
    # F_bandpower from P
    
    Cov_bandpower_PP = CombineCovariance3(l, matricesPP)
    #print Cov_bandpower_PP.shape
    
    
    DAT1 = np.vstack(( karray, Cov_bandpower_PP ))
    
    # GET full C_Xi
    l_r = len(RSDPower.rcenter)
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    
    # F_bandpower from Xi
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    Cov_bandpower_Xi = pinv( Fisher_bandpower_Xi, rcond=1e-15 )
    
    np.allclose(Fisher_bandpower_Xi, np.dot(Fisher_bandpower_Xi, np.dot(Cov_bandpower_Xi, Fisher_bandpower_Xi)))
    
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
    DAT2 = np.vstack(( karray, Cov_bandpower_Xi ))
    DAT3 = np.vstack(( karray, Fisher_bandpower_Xi ))

    # data vector P
    #data_Vec = np.array([RSDPower.multipole_bandpower0[0:l+1], RSDPower.multipole_bandpower2[0:l+1], RSDPower.multipole_bandpower4[0:l+1]]).reshape(1,3 * (l+1))
    DATVEC = np.column_stack(( RSDPower.kcenter, RSDPower.multipole_bandpower0, RSDPower.multipole_bandpower2, RSDPower.multipole_bandpower4 ))
    
    
    
    np.savetxt('covP'+suffix+'.txt', DAT1, delimiter = ' ', header = '#\nCovPP', comments = '# first column is k'  )
    np.savetxt('covXi'+suffix+'.txt', DAT2, delimiter = ' ', header = '#\nCovPP from Xi', comments = '# first column is k'  )
    np.savetxt('FisherXi'+suffix+'.txt', DAT3, delimiter = ' ', header = '#\nFisherPP from Xi', comments = '# first column is k'  )
    np.savetxt('dataVector'+suffix+'.txt', DATVEC , delimiter = ' ', header = ' k       P0        P2       P4', comments = '#'  )




def convergence_Xi(RSDPower):

    kcut_min = get_closest_index_in_data( .0, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( 200, RSDPower.kmax )
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    """
    matrices2Xi = [RSDPower.dxip0
                 , Xizeros
                 , Xizeros
                 , Xizeros
                 , RSDPower.dxip2
                 , Xizeros
                 , Xizeros
                 , Xizeros
                 , RSDPower.dxip4]
    """
    
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

    cut = kcut_max + 1- kcut_min  #len(RSDPower.kcenter)
    part00 = Cov_bandpower_Xi[0:cut, 0:cut]
    part02 = Cov_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Cov_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Cov_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Cov_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Cov_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]

    part_list = [ part00, part02, part04, np.transpose(part02), part22, part24, np.transpose(part04), np.transpose(part24), part44]
    
    
    SNR_Xi_list = []
    for l in range(50):
    
        Cov_bandpower_Xi_combine = CombineCovariance3(l, part_list)
        part_Fisher_bandpower_Xi = inv( Cov_bandpower_Xi_combine ) #, rcond = 1e-15)

        data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).reshape(1,3 * multipole_P0[0:l+1].size)

        SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), np.transpose(data_Vec))
        print RSDPower.kcenter[kcut_min:kcut_max+1][l], SNR_Xi
        SNR_Xi_list.append(SNR_Xi)

    return [RSDPower.kcenter[kcut_min:kcut_max+1], SNR_Xi_list]

def convergence_P(RSDPower):
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
    
    
    kcut_min = get_closest_index_in_data( .0, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( 20, RSDPower.kmax )
    
    #l = kcut_max - kcut_min


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
        
    SNR_PP_list = []
    for l in range((kcut_max - kcut_min)):
        part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
        data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).reshape(1,3 * multipole_P0[0:l+1].size)
    
        SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
        print SNR_PP
        SNR_PP_list.append(SNR_PP)
    
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


    #KMIN = 0.1
    #KMAX = 20. #502.32
    #kN= P0.size + 1
    #kbin, dk = np.linspace(KMIN, KMAX, 2**11+1, retstep = True)
    #kmin = np.delete(kbin,-1)
    #kcenter = kmin + dk/2.

    #return kcenter[0:kcenter.size/2], SNR_PP_list, SNR_Xi_list
    return RSDPower.kcenter[0:(kcut_max - kcut_min)], SNR_PP_list


    """
    fig, (ax, ax2, ax3) = plt.subplots(1,3)
    ax3.plot(kcenter, SNRPP, 'r.')
    ax3.plot(kcenter2, SNRPP2, 'b.')
    ax3.plot(kcenter3, SNRPP3, 'g.')

    """


def Cumulative_SNR_loop(RSDPower, l):
    """
    Calculate cumulative Signal to Noise up to kmax
    
    Parameter
    ---------
    RSDPower: class name
    l: slicing index. determin kmax=k[l]
    
    """
    kcut_min = get_closest_index_in_data( .0, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( 20, RSDPower.kmax )

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
    Cov_bandpower_Xi = pinv( Fisher_bandpower_Xi ) #, rcond = 1e-15 )
    
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
    part_Fisher_bandpower_Xi = pinv( Cov_bandpower_Xi_combine ) #, rcond = 1e-15)
    
    #a = Cov_bandpower_Xi_combine
    #B = part_Fisher_bandpower_Xi
    #check = np.allclose(np.dot(a, B), np.eye(a.shape[0]))
    #check = np.allclose(a, np.dot(a, np.dot(B, a)))
    #print check, '3rd'
    
    data_Vec = np.array([multipole_P0[0:l+1], multipole_P2[0:l+1], multipole_P4[0:l+1]]).reshape(1,3 * multipole_P0[0:l+1].size)
    
    data_Vec0 = RSDPower.multipole_bandpower0[0:l+1]
    Fisher_bandpower_PP00 = inv(RSDPower.covariance_PP00[0:l+1,0:l+1])
    
    SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
    SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), np.transpose(data_Vec))
    
    return RSDPower.kcenter[kcut_min:kcut_max+1][l], SNR_PP, SNR_Xi





def SNR_multiprocessing(RSDPower, kcut_max):
    """
    Do multiprocessing for the function Cumulative_SNR_loop()
    
    Parameter
    ---------
    RSDPower: class name
    kcut_max: index of kmax
    
    """
    num_process = 8
    
    #print 'multi_processing for k loop : ', num_process, ' workers'
    numberlist_k = np.arange(1, kcut_max-1,1)
    numberlist_k_split = np.array_split(numberlist_k, num_process)
    
    vec_Cumulative_SNR_loop = np.vectorize(Cumulative_SNR_loop)
    
    def cumulative_SNR_all(q, order, input):
    
        kklist, SNR_PP_list, SNR_Xi_list = vec_Cumulative_SNR_loop(RSDPower,input)
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


    """
    fig, ax = plt.subplots()
    ax.plot(kklist4, SNRPP_list4, 'r+', label = 'SNR P')
    ax.plot(kklist2, SNR_Xi_list, 'b+', label = 'SNR Xi')
    ax.set_xlim(0,2)
    ax.set_title('Signal To Noise, evenly spaced, k:1e-4~200, r:30~200')
    ax.set_ylabel('(S/N)^2')
    ax.set_xlabel('k (h/Mpc)')

    fig.savefig('SNR_even')
    """

def main( ):

    # initial setting ------------------------------
    
    #  (parameter description in class code)
    KMIN = 0.0001
    KMAX = 100. #502.32
    RMIN = 30.
    RMAX = 200.
    kmin = .0
    kmax = 20
    
    # the number of k sample point should be 2^n+1 (b/c romb integration)
    kN = 2**6 + 1
    rN = 401
    subN = 2**10 + 1
    N_x = 2**13 + 1
    
    # RSD class
    RSDPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN, N_x, logscale = False)
    #LogPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN, N_x, logscale = True)
    #RSDPower.compile_fortran_modules() # compile only one time
    
    rcut_max = len(RSDPower.rcenter)-1
    rcut_min = 0
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax )

    print "\nkcut_min :", RSDPower.kmin[kcut_min], "  kcut_max :", RSDPower.kmax[kcut_max]
    print "rcut_min :", RSDPower.rmin[rcut_min], "  rcut_max :", RSDPower.rmax[rcut_max],"\n"




    # make bandpower and cov matrices-----------------------
    # (function description in class code)
    
    file = 'matterpower_z_0.55.dat' # from camb (z=0.55)
    RSDPower.MatterPower(file = file)
    #RSDPower.Shell_avg_band()
    
    print '\nStarting multiprocessing (about 60 sec for test code)'
    # power spectrum multipoles l = 0,2,4
    RSDPower.multipole_P_band_all()
    
    # derivative dXi/dp
    #RSDPower.derivative_Xi_band_all()

    # P covariance matrix ( nine submatrices C_ll' )
    RSDPower.RSDband_covariance_PP_all()

    # Xi covariance matrix ( nine submatrices C_ll' )
    #RSDPower.RSDband_covariance_Xi_all()
    
    #suffix = '_kbin'+str(kN)+'_ps'
    #DataSave(RSDPower, suffix = suffix)
    #kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(RSDPower, kcut_max)
    
    kklist, SNRPP = convergence_P(RSDPower)
    print SNRPP
    return kklist, SNRPP
    
    
    print SNR_PP_list
    print SNR_Xi_list
    return kklist, SNR_PP_list, SNR_Xi_list
    stop
    """
    
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    
    matrices2Xi = [RSDPower.dxip0
        , Xizeros
        , Xizeros
        , Xizeros
        , RSDPower.dxip2
        , Xizeros
        , Xizeros
        , Xizeros
        , RSDPower.dxip4]
        
    
    multipole_P0, multipole_P2, multipole_P4 = RSDPower.multipole_bandpower0, RSDPower.multipole_bandpower2, RSDPower.multipole_bandpower4

    # GET full C_Xi
    l_r = len(RSDPower.rcenter)
    C_matrix3 = CombineCovariance3(l_r, matricesXi)

    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)  # Here, cut out small and large k

    # F_bandpower from Xi
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    
    
    #Cov_bandpower_Xi = inv( Fisher_bandpower_Xi )
    data_Vec = np.array([multipole_P0, multipole_P2, multipole_P4]).ravel()
    
    SNR = np.dot( np.dot( data_Vec, Fisher_bandpower_Xi), data_Vec.T)
                   
    return SNR
    """
    # -------------------------------
    rcenter, kcenter, P0, dxip0, covariance = RSDPower.rcenter, RSDPower.kcenter, RSDPower.multipole_bandpower0, RSDPower.dxip0, RSDPower.covariance00
    
    FisherP = np.dot( np.dot( dxip0, np.linalg.inv(covariance)), dxip0.T)
    
    SNRLIST=[]
    for l in range(P0.size):
        SNR = np.dot( np.dot (P0[0:l+1], FisherP[0:l+1, 0:l+1]), P0[0:l+1].T )
        SNRLIST.append(SNR)
    SNRLIST = np.array(SNRLIST).ravel()
    #print SNRLIST
    #return [RSDPower.kcenter, SNRLIST]
    return RSDPower.rcenter, RSDPower.kcenter, RSDPower.multipole_bandpower0, RSDPower.dxip0, RSDPower.covariance00, SNRLIST
    
    
    #return RSDPower.rcenter, RSDPower.covariance00
    
    results = []
    for l in range(30):
        result = Cumulative_SNR_loop(RSDPower, l)
        results.append(result)
        print result
    
    return results

    results = np.array(result).ravel()
    results = result.reshape(50, 3)

    return results


    fig, ax = plt.subplots()
    ax.plot(results[:,0], results[:,1], 'r3', label='kbin 11, 10')
    ax.plot(results[:,0], results[:,2], 'b3', label='kbin 11, 10')
    
    
    #stop
    
    kcenter, SNR_PP = convergence_P(RSDPower)
    SNR_PP = np.array(SNR_PP).ravel()
    return kcenter, SNR_PP
    #print SNR_PP
    
    # LogPower
    # (function description in class code)
    """
    file = 'matterpower_z_0.55.dat' # from camb (z=0.55)
    LogPower.MatterPower(file =file)
    #RSDPower.Shell_avg_band()
    
    print '\nStarting multiprocessing (about 60 sec for test code)'
    # power spectrum multipoles l = 0,2,4
    LogPower.multipole_P_band_all()
    
    # derivative dXi/dp
    LogPower.derivative_Xi_band_all()
    
    # P covariance matrix ( nine submatrices C_ll' )
    LogPower.RSDband_covariance_PP_all()
    
    # Xi covariance matrix ( nine submatrices C_ll' )
    LogPower.RSDband_covariance_Xi_all()
    """
    #suffix = '_kbin'+str(kN)+'_ps'
    #DataSave(RSDPower, suffix = suffix)
    
    #stop
    """
    Log_result1, Log_result2, Log_result3, Log_result4 = LogPower.RSDband_covariance_PP(0,0)
    Even_result1, Even_result2,Even_result3,Even_result4 = RSDPower.RSDband_covariance_PP(0,0)
    Log_V =  LogPower.RSDband_covariance_PP(0,0)
    Even_V =  RSDPower.RSDband_covariance_PP(0,0)
    
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2)

    ax3.plot(kklist, SNR_PP_list, 'r.', label = 'P')
    ax4.plot(kklist, SNR_Xi_list, 'r.', label = 'Xi')

    ax.loglog(LogPower.kcenter, Log_V, 'r-', label = 'log')
    ax.loglog(RSDPower.kcenter, Even_V, 'b.', label = 'even')

    ax.loglog(LogPower.kcenter, Log_result1, 'r-', label = 'log')
    ax.loglog(RSDPower.kcenter, Even_result1, 'b.', label = 'even')
    ax2.loglog(LogPower.kcenter, Log_result2, 'r-', label = 'log')
    ax2.loglog(RSDPower.kcenter, Even_result2, 'b.', label = 'even')
    ax3.loglog(LogPower.kcenter, Log_result3, 'r-', label = 'log')
    ax3.loglog(RSDPower.kcenter, Even_result3, 'b.', label = 'even')
    ax4.loglog(LogPower.kcenter, Log_result4, 'r-', label = 'log')
    ax4.loglog(RSDPower.kcenter, Even_result4, 'b.', label = 'even')
    
    ax5.loglog(LogPower.kcenter, LogPower.covariance_PP00.diagonal(), 'r-')
    ax5.loglog(RSDPower.kcenter, RSDPower.covariance_PP00.diagonal(), 'b.')
    ax5.legend(loc = 'best')
    ax5.legend(loc='best')
    fig, (ax, ax2) = plt.subplots(1,2)
    im = ax.imshow(LogPower.covariance_PP00)
    im2 = ax2.imshow(RSDPower.covariance_PP00)
    cbar = fig.colorbar(im, ax = ax)
    cbar2 = fig.colorbar(im2, ax = ax2)
    """
    
    # cumulative SNR and plotting ----------------------------

    #kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(RSDPower, kcut_max)
    #kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(LogPower, kcut_max_l)
    """
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( 200, RSDPower.kmax )
    ll = np.arange(kcut_max + 1 - kcut_min)
    
    kklist = []
    SNR_PP_list = []
    SNR_Xi_list = []
    for l in ll:
        kk, SNR_PP, SNR_Xi = Cumulative_SNR_loop(RSDPower, l)
        kklist.append(kk)
        SNR_PP_list.append(SNR_PP)
        SNR_Xi_list.append(SNR_Xi)
        #print SNR_PP

    kklist = np.array(kklist).ravel()
    SNR_PP_list = np.array(SNR_PP_list).ravel()
    SNR_Xi_list = np.array(SNR_Xi_list).ravel()

    print SNR_PP_list
    print SNR_Xi_list
    """
    #print SNR_PP_list
    #print SNR_Xi_list
    #print kklist
    #return kklist, SNR_PP_list, SNR_Xi_list


def main2():
    
    results = []
    for i in range(11,15):
        kn = 2 **i + 1
        result = main( kN = kn )
        results.append(result)

    print results

    stop
    main( kN = kn )
    main( kN = kn )
    stop
    
    kklist, SNR_PP_list, SNR_Xi_list = main( kN = 2**6 + 1 )
    kklist2, SNR_PP_list2, SNR_Xi_list2 = main( kN = 2**7 + 1 )
    kklist3, SNR_PP_list3, SNR_Xi_list3 = main( kN = 2**8 + 1 )
    #kklist4, SNR_PP_list4, SNR_Xi_list4 = main( kN = 2**8 + 1 )
    
    fig, (ax, ax2) = plt.subplots(1,2, figsize= (14, 7))
    ax.plot(kklist, SNR_PP_list, 'b.', label = '10' )
    ax.plot(kklist2, SNR_PP_list2, 'r.', label = '11' )
    ax.plot(kklist3, SNR_PP_list3, 'g.', label = '9')
    #ax.plot(kklist4, SNR_PP_list4, 'o', label = 'kbin257', alpha = 0.3  )
    
    ax2.plot(kklist, SNR_Xi_list, 'b.', label = '10' )
    ax2.plot(kklist2, SNR_Xi_list2, 'r.', label = '11' )
    ax2.plot(kklist3, SNR_Xi_list3, 'g.', label = '9')
    #ax2.plot(kklist4, SNR_Xi_list4, 'o', label = 'kbin257', alpha = 0.3 )

    ax.set_xlim(0,20)
    ax.legend(loc = 'best')
    ax.set_title('Power Spectrum')
    ax2.set_xlim(0,20)
    ax2.set_ylim(0, 2e5)
    #ax2.set_yscale('log')
    #ax2.set_ylabel('Cumulative (S/N)^2')
    #ax2.set_xlabel('k (h/Mpc)')
    ax2.legend(loc = 'best')
    ax2.set_title('Correlation Function')
    #ax.set_xscale('log')
    fig.suptitle('k = [0.1, 20] h/Mpc, r = [10, 200] Mpc/h, log-spaced')
    fig.savefig('test')

    
    makedirectory('plots')
    # make plot
    Linear_plot( kklist, ['P','Xi'], SNR_PP_list, SNR_Xi_list, scale = None, title = 'Cumulative SNR \n (rmin : {:>3.3f} rmax : {:>3.3f})'.format(RSDPower.RMIN, RSDPower.RMAX), pdfname = 'plots/cumulative_snr_kN{}_rN{}_rmin{:>3.3f}_rmax{:>3.3f}.png'.format(RSDPower.n, RSDPower.n2, RSDPower.RMIN, RSDPower.RMAX), ymin = 0.0, ymax = 1e7, xmin = 0.0, xmax = 2, ylabel='Cumulative SNR' )













if __name__=='__main__':
    main2()
