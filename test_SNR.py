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
/    Cov P(k)
---
k_min


* Cov P_{xi}

d Xi              d Xi
=  ---- [Cov P]^(-1) ----
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
import matplotlib.pyplot as plt
from error_analysis_class import *


def Cumulative_SNR_loop(RSDPower, l):
    """
    Calculate cumulative Signal to Noise up to kmax
    
    Parameter
    ---------
    RSDPower: class name
    l: slicing index. determin kmax=k[l]
    
    """
    
    matricesXi = [RSDPower.covariance00, RSDPower.covariance02, RSDPower.covariance04, np.transpose(RSDPower.covariance02), RSDPower.covariance22, RSDPower.covariance24,np.transpose(RSDPower.covariance04), np.transpose(RSDPower.covariance24), RSDPower.covariance44]
    
    matricesPP = [RSDPower.covariance_PP00, RSDPower.covariance_PP02, RSDPower.covariance_PP04,RSDPower.covariance_PP02, RSDPower.covariance_PP22, RSDPower.covariance_PP24,RSDPower.covariance_PP04, RSDPower.covariance_PP24, RSDPower.covariance_PP44]
    
    Xizeros = np.zeros((len(RSDPower.kcenter),len(RSDPower.rcenter)))
    matrices2Xi = [RSDPower.dxip0, Xizeros,Xizeros,Xizeros,RSDPower.dxip2,Xizeros,Xizeros,Xizeros,RSDPower.dxip4]
    

    # F_bandpower from P
    part_Fisher_bandpower_PP = inv(CombineCovariance3(l, matricesPP))
    
    # GET full C_Xi
    l_r = len(RSDPower.rcenter)
    C_matrix3 = CombineCovariance3(l_r, matricesXi)
    
    Xi, Xi2 = CombineDevXi3(l_r, matrices2Xi)
    
    # F_bandpower from Xi
    Fisher_bandpower_Xi = FisherProjection(Xi, C_matrix3)
    Cov_bandpower_Xi = inv( Fisher_bandpower_Xi )
    
    
    cut = len(RSDPower.kcenter)
    
    part00 = Cov_bandpower_Xi[0:cut, 0:cut]
    part02 = Cov_bandpower_Xi[0:cut, cut:2*cut]
    part04 = Cov_bandpower_Xi[0:cut, 2*cut:3*cut+1]
    part22 = Cov_bandpower_Xi[cut:2*cut, cut:2*cut]
    part24 = Cov_bandpower_Xi[cut:2*cut, 2*cut:3*cut+1]
    part44 = Cov_bandpower_Xi[2*cut:3*cut+1, 2*cut:3*cut+1]
    
    part_list = [ part00, part02, part04, np.transpose(part02), part22, part24, np.transpose(part04), np.transpose(part24), part44]
    Cov_bandpower_Xi_combine = CombineCovariance3(l, part_list)
    part_Fisher_bandpower_Xi = inv(Cov_bandpower_Xi_combine )
    
    data_Vec = np.array([RSDPower.multipole_bandpower0[0:l+1], RSDPower.multipole_bandpower2[0:l+1], RSDPower.multipole_bandpower4[0:l+1]]).reshape(1,3 * (l+1))
    
    data_Vec0 = RSDPower.multipole_bandpower0[0:l+1]
    Fisher_bandpower_PP00 = inv(RSDPower.covariance_PP00[0:l+1,0:l+1])
    
    SNR_PP = np.dot( np.dot( data_Vec, part_Fisher_bandpower_PP ), np.transpose(data_Vec))
    SNR_Xi = np.dot( np.dot( data_Vec, part_Fisher_bandpower_Xi ), np.transpose(data_Vec))
    
    return RSDPower.kcenter[l], SNR_PP, SNR_Xi



def SNR_multiprocessing(RSDPower, kcut_max):
    """
    Do multiprocessing for the function Cumulative_SNR_loop()
    
    Parameter
    ---------
    RSDPower: class name
    kcut_max: index of kmax
    
    """
    num_process = 4
    
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
    SNR_PP_list = np.sqrt(loops[1])
    SNR_Xi_list = np.sqrt(loops[2])
    
    #print "done \nSNR final 10 values :", SNR_Xi_list[-11:-1]
    print 'done'
    return kklist, SNR_PP_list, SNR_Xi_list




def main():

    
    # initial setting ------------------------------
    
    #  (parameter description in class code)
    KMIN = 0.01
    KMAX = 200 #502.32
    RMIN = .1
    RMAX = 200.
    kmin = KMIN
    kmax = KMAX
    
    # the number of k sample point should be 2^n+1 (b/c romb integration)
    kN = 2**6 + 1
    rN = 201
    subN = 2**11 + 1
    N_x = 2**14 + 1
    
    # RSD class
    RSDPower = RSD_covariance(KMIN, KMAX, RMIN, RMAX, kN, rN, subN, N_x)
    RSDPower.compile_fortran_modules() # compile only one time
    
    rcut_max = len(RSDPower.rcenter)-1
    rcut_min = 0
    kcut_min = get_closest_index_in_data( kmin, RSDPower.kmin )
    kcut_max = get_closest_index_in_data( kmax, RSDPower.kmax )
    
    print "\nkcut_min :", RSDPower.kmin[kcut_min], "  kcut_max :", RSDPower.kmax[kcut_max]
    print "rcut_min :", RSDPower.rmax[rcut_min], "  rcut_max :", RSDPower.rmin[rcut_max],"\n"
    
    
    
    # make bandpower and cov matrices-----------------------
    # (function description in class code)
    
    file = open('matterpower_z_0.55.dat') # from camb (z=0.55)
    RSDPower.MatterPower(file)
    #RSDPower.Shell_avg_band()
    
    print 'Starting multiprocessing (about 60 sec for test code)'
    # power spectrum multipoles l = 0,2,4 """
    RSDPower.multipole_P_band_all()
    
    # derivative dXi/dp """
    RSDPower.derivative_Xi_band_all()

    # P covariance matrix ( nine submatrices C_ll' ) """
    RSDPower.RSDband_covariance_PP_all()

    # Xi covariance matrix ( nine submatrices C_ll' ) """
    RSDPower.RSDband_covariance_Xi_all()
    

    
    # cumulative SNR and plotting ----------------------------

    kklist, SNR_PP_list, SNR_Xi_list = SNR_multiprocessing(RSDPower, kcut_max)
    
    makedirectory('plots')
    # make plot
    Linear_plot( kklist, ['P','Xi'], SNR_PP_list, SNR_Xi_list, scale = None, title = 'Cumulative SNR \n (rmin : {:>3.3f} rmax : {:>3.3f})'.format(RSDPower.RMIN, RSDPower.RMAX), pdfname = 'plots/cumulative_snr_kN{}_rN{}_rmin{:>3.3f}_rmax{:>3.3f}.pdf'.format(RSDPower.n, RSDPower.n2, RSDPower.RMIN, RSDPower.RMAX), ymin = 0.0, ymax = 800., xmin = 0.0, xmax = 1., ylabel='Cumulative SNR' )


if __name__=='__main__':
    main()
