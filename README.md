
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

