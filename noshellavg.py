import numpy as np
from numpy import zeros, sqrt, pi, sin, cos, exp
from numpy.linalg import pinv as inv
from numpy import vectorize
from scipy.interpolate import interp1d
#from scipy.integrate import simps
import sys
import matplotlib.pyplot as plt
from scipy_integrate import *

def Pmultipole(l, binavg = True):
    """
    Vs= 5.0*10**9
    nn= 3.0 * 10**(-4)
    
    KMIN = 0.01
    KMAX = 200. #502.32
    RMIN = 29.
    RMAX = 200.
    kmin = .1
    kmax = 50.
    
    # the number of k sample point should be 2^n+1 (b/c romb integration)
    #kN = 2**10 + 1
    kN = 100
    rN = 31
    subN = 2**6 + 1
    # bin

    b=2.0
    f=0.74
    s= 3.5
        
    Nmu = 2**6 + 1

    
    mulist, dmu = np.linspace(-1.,1., Nmu, retstep = True)
    
    kbin, dk = np.linspace(KMIN, KMAX, kN, retstep = True)
    kmin = np.delete(kbin,-1)
    kmax = np.delete(kbin,0)
    kcenter = kmin + dk/2.
    skbin = np.linspace(KMIN, KMAX, subN * kN)

    file ='matterpower_z_0.55.dat'
    fo = open(file, 'r')
    position = fo.seek(0, 0)
    Pkl=np.array(np.loadtxt(fo))
    k=np.array(Pkl[:,0])
    P=np.array(Pkl[:,1])

    Pm = interp1d(k, P, kind= "cubic")
    #self.Pmlist = Pm(self.kcenter)

    #REAL POWERSPECTRUM DATA
    PS = np.array([Pm(sk) for sk in skbin])
    """

    Vs, nn, b, f, s, kN, rN, subN, Nmu, KMIN, KMAX, RMIN, RMAX, mulist, dmu, kbin, dk, kmin, kmax, kcenter, skbin, sdk, rbin, dr, rmin, rmax, rcenter, srbin, PS = InitialCondi()

    # multipole binning

    matrix1, matrix2 = np.mgrid[0:mulist.size,0:skbin.size]
    mumatrix = mulist[matrix1]
    Le_matrix = Ll(l,mumatrix)
    
    #Vi = 4./3 * pi * (kmax**3 - kmin**3)
    Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
    #Nk = Vs * Vi /2/(2*pi)*

    kmatrix = skbin[matrix2]
    Pmmatrix = PS[matrix2]
    Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * s**2)
    R = (b + f * mumatrix**2)**2 * Dmatrix * Le_matrix
    int = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * dk * Pmmatrix * R, dx = dmu, axis=0 )
    #int = (2 * l + 1.)/2 * simps( Pmmatrix * R, dx = dmu, axis=0 )

    # binning

    if binavg == True:
        Inds = np.digitize( skbin, kbin )
        Pmultipole = [np.sum(int[Inds == i])/len(int[Inds == i]) for i in range(1,kbin.size)]
        #Pmultipole = [np.sum(int[Inds == i]) for i in range(1,kbin.size)]
        Pmultipole = np.array(Pmultipole) /Vi
    
    elif binavg == False:
        Pmultipole = (2 * l + 1.)/2 * simps( Pmmatrix * R, dx = dmu, axis=0 )

    return Pmultipole



def _Covariance_PP(l1, l2, binavg = True):

    Vs, nn, b, f, s, kN, rN, subN, Nmu, KMIN, KMAX, RMIN, RMAX, mulist, dmu, kbin, dk, kmin, kmax, kcenter, skbin, sdk, rbin, dr, rmin, rmax, rcenter, srbin, PS = InitialCondi()

    # FirstTerm + SecondTerm
    matrix1, matrix2 = np.mgrid[0:mulist.size,0:skbin.size]
    mumatrix = mulist[matrix1]
        
    Le_matrix1 = Ll(l1,mumatrix)
    Le_matrix2 = Ll(l2,mumatrix)
    Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
    #Vi = 4./3 * pi * ( kmax**3 - kmin**3 )
    Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) * (2*pi)**3 /Vs
    
    
    k = skbin
    kmatrix = skbin[matrix2]
    Pmmatrix = PS[matrix2]
    Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * s**2) #FOG matrix
    R = (b + f * mumatrix**2)**2 * Dmatrix
    Rintegral3 = simps(  R**2 * Le_matrix1 * Le_matrix2, dx = dmu, axis=0 )
    Rintegral2 =  simps(  R * Le_matrix1 * Le_matrix2, dx = dmu, axis=0 )
    int1 = 4 * pi * k**2 * dk * PS**2 * Rintegral3
    int2 = 4 * pi * k**2 * dk * PS * Rintegral2
    #int1 = PS**2 * Rintegral3
    #int2 = PS * Rintegral2
    

    Inds = np.digitize( skbin, kbin )
    First = [np.sum(int1[Inds == i])/len(int1[Inds == i]) for i in range(1,kbin.size)]
    Second = [np.sum(int2[Inds == i])/len(int2[Inds == i]) for i in range(1,kbin.size)]
    #First = [np.sum(int1[Inds == i]) for i in range(1,kbin.size)]
    #Second = [np.sum(int2[Inds == i]) for i in range(1,kbin.size)]

    FirstTerm = Const_alpha * np.array(First)/Vi**2
    SecondTerm = Const_alpha * 2./nn * np.array(Second)/Vi**2
    
    NAvgFirstTerm = Const_alpha * PS**2 * Rintegral3
    NAvgSecondTerm = Const_alpha * 2./nn * PS * Rintegral2
    
    # LastTerm
        
    if l1 == l2:
        LastTerm = (2*l1 + 1.) * 2. * (2 * pi)**3/Vs/nn**2 / Vi
        NAvgLast = (2*l1 + 1.) * ( 2 * pi)**2 / Vs / nn**2 / skbin**2
    else:
        LastTerm = 0.
        NAvgLast = 0.

    if binavg == True :
        Total = FirstTerm + SecondTerm + LastTerm
        #Total = LastTerm
        covariance_mutipole_PP = np.zeros((len(kcenter),len(kcenter)))
        np.fill_diagonal(covariance_mutipole_PP,Total)

    elif binavg == False:
        NavgTotal = NAvgFirstTerm + NAvgSecondTerm + NAvgLast
        covariance_mutipole_PP = np.zeros((len(skbin),len(skbin)))
        np.fill_diagonal(covariance_mutipole_PP,NavgTotal)

    #return NAvgFirstTerm + NAvgSecondTerm + NAvgLast
    return covariance_mutipole_PP



def _derivative_Xi(l):

    import numpy as np
    from numpy import pi
    import cmath
    I = cmath.sqrt(-1)


    Vs, nn, b, f, s, kN, rN, subN, Nmu, KMIN, KMAX, RMIN, RMAX, mulist, dmu, kbin, dk, kmin, kmax, kcenter, skbin, sdk, rbin, dr, rmin, rmax, rcenter, srbin, PS = InitialCondi()

    
    matrix1, matrix2 = np.mgrid[0:rcenter.size,0:skbin.size]
    
    k = skbin
    kmatrix = skbin[matrix2]
    rminmatrix = rmin[matrix1]
    rmaxmatrix = rmax[matrix1]
    Vir = 4./3 * pi * np.fabs(rmaxmatrix**3 - rminmatrix**3)
    AvgBesselmatrix = avgBessel(l, kmatrix ,rminmatrix, rmaxmatrix) /Vir
    int = np.real(I**l) * 4 * pi * kmatrix**2 * dk * AvgBesselmatrix /(2 * pi)**3

    Inds = np.digitize( skbin, kbin )
    result = [np.sum(int[j][Inds == i])/len(int[j][Inds == i]) for i in range(1,kbin.size) for j in range(rbin.size-1)]
    result = np.array(result).reshape(rcenter.size, kcenter.size)
    
    #sys.stdout.write('.')
    
    return result



def _covariance_Xi(l1, l2):
    from numpy import pi, real
    from scipy.integrate import simps
    import cmath
    
    I = cmath.sqrt(-1)

    Vs, nn, b, f, s, kN, rN, subN, Nmu, KMIN, KMAX, RMIN, RMAX, mulist, dmu, kbin, dk, kmin, kmax, kcenter, skbin, sdk, rbin, dr, rmin, rmax, rcenter, srbin, PS = InitialCondi()

    KMIN = 0.01
    KMAX = 200
    kbin, dk = np.linspace(KMIN, KMAX, kN, retstep = True)
    kmin = np.delete(kbin,-1)
    kmax = np.delete(kbin,0)
    kcenter = kmin + dk/2.
    skbin, sdk = np.linspace( KMIN, KMAX, subN, retstep = True )
    
    const_gamma = np.real(I**(l1+l2)) * 2.* (2*l1+1)*(2*l2+1) /(2*pi)**2 /Vs
    
    matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skbin)]
    mumatrix = mulist[matrix1] # mu matrix (axis 0)
    kmatrix = skbin[matrix2] # k matrix (axis 1)
    Pmmatrix = PS[matrix2]
    Le_matrix1 = Ll(l1,mumatrix)
    Le_matrix2 = Ll(l2,mumatrix)
    
    Dmatrix = np.exp(-kmatrix**2 * mumatrix**2 * s**2)
    R = (b + f * mumatrix**2)**2 * Dmatrix

    Rintegral3 = simps(R**2 * Le_matrix1 * Le_matrix2, mulist, axis=0 )
    Rintegral2 = simps(R * Le_matrix1 * Le_matrix2, mulist, axis=0 )
    result = const_gamma * skbin**2 * (Rintegral3 * PS**2 + Rintegral2 * PS * 2./nn)

    Vi = 4./3 * pi * np.fabs(rmin**3 - rmax**3)

    FirstSecond = []
    
    for j in range(rmin.size):
        for i in range(rmin.size):
            AvgBessel1 = avgBessel(l1,skbin,rmin[i],rmax[i])/Vi[i]
            AvgBessel2 = avgBessel(l2,skbin,rmin[j],rmax[j])/Vi[j]
            re = simps(result * AvgBessel1 * AvgBessel2, skbin)
            FirstSecond.append(re)

    FirstSecond = np.array(FirstSecond).reshape(rmin.size, rmin.size)

    if l1 == l2:
        Last = (2./Vs) * (2*l1+1)/nn**2 / Vi
        LastTerm = np.zeros((len(rcenter),len(rcenter)))
        np.fill_diagonal(LastTerm,Last)
    else : LastTerm = 0.

    covariance = FirstSecond + LastTerm
    return covariance



def get_closest_index_in_data( value, data ):

    for i in range(len(data)):
        if data[i] < value : pass
        elif data[i] >= value :
            if np.fabs(value - data[i]) > np.fabs(value - data[i-1]):
                value_index = i-1
            else : value_index = i
            break


    if data[0] > data[1]:
        for i in range(len(data)):
            if data[i] > value : pass
            elif data[i] <= value :
                if np.fabs(value - data[i]) > np.fabs(value - data[i-1]): value_index = i-1
                else : value_index = i
                break

        if value_index == -1: value_index = data.size-1

    return value_index

def Ll(l,x):
    
    """ 
    Calculate Legendre Polynomial function L_l (x)
    
    Parameters
    ----------
    l: order (0,2,4)
    x: input number
    
    """
    
    import numpy as np
    from numpy import vectorize
    from fortranfunction import eval_legendre
    
    Le_func = lambda lp,xp : eval_legendre(lp,xp)
    Le_func = np.vectorize(Le_func)
    result = Le_func(l,x)
    
    return result


def _avgBessel(l,k,rmin,rmax):
    
    """ 
    Calculate Averaged spherical Bessel function J_l (kr) in configuration space in each bin
    * fortranfunction module needed
    
    Parameters
    ----------
    l: order of spherical Bessel function (0,2,4)
    k: center value in each k bin
    rmin, rmax: minimum and maximum r values in each r bin
    
    """
    
    from numpy import vectorize, pi, cos, sin
    from fortranfunction import sici
    sici = vectorize(sici)

    
    if l == 0 :
        result = 4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin))/(k**3)
    elif l == 2 :
        result = 4. * pi * (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax) - 3*sici(k*rmin))/k**3
    else :
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * cos(k * rmax) +\
                                  (- 105 * k/rmin + 2 * k**3 * rmin) * cos(k * rmin) +\
                                  22 * k**2 * sin(k * rmax) - (105 * sin(k * rmax))/rmax**2 -\
                                  22 * k**2 * sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                                  15 * k**2 * (sici(k * rmax) - sici(k * rmin)))
    return result




def CombineDevXi(l, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """

    dxib0 = matrices[0][0:l+1]
    dxib2 = matrices[1][0:l+1]
    dxib4 = matrices[2][0:l+1]
    dxif0 = matrices[3][0:l+1]
    dxif2 = matrices[4][0:l+1]
    dxif4 = matrices[5][0:l+1]
    dxis0 = matrices[6][0:l+1]
    dxis2 = matrices[7][0:l+1]
    dxis4 = matrices[8][0:l+1]
    
    Matrix1 = np.array([dxib0, dxib2, dxib4]).ravel()
    Matrix2 = np.array([dxif0, dxif2, dxif4]).ravel()
    Matrix3 = np.array([dxis0, dxis2, dxis4]).ravel()
    Xi = np.vstack((Matrix1, Matrix2, Matrix3))

    Matrix1 = np.array([dxib0, dxib2]).ravel()
    Matrix2 = np.array([dxif0, dxif2]).ravel()
    Matrix3 = np.array([dxis0, dxis2]).ravel()
    Xi2 = np.vstack((Matrix1, Matrix2, Matrix3))
    
    return Xi, Xi2



def confidence_ellipse(x_center, y_center, linestyle, linecolor, *args):
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import figure, show, rand
    from matplotlib.patches import Ellipse
    
    # For BAO and RSDscales

    if linecolor == None : linecolor = ['b', 'r', 'g', 'b', 'r', 'g', 'y', 'c', 'k']
    if linestyle == None : linestyle = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid']
    ziplist = zip(args, linecolor, linestyle)
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    elllist = []
    
    
    for z in ziplist:
        vals, vecs = eigsorted(z[0])
        #print "values :", vals
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        nstd = np.sqrt(5.991) # 95% :4.605  #99% :9.210
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(x_center, y_center),
              width=w, height=h,
              angle=theta, color = z[1], ls = z[2], lw=1.5, fc= 'None')
        elllist.append(ell)
        
    return elllist

    """
    for e in elllist:
        ax.add_artist(e)
        #e.set_alpha(0.2)
        e.set_clip_box(ax.bbox)


    xmin =  x_center*0.97
    xmax =  x_center*1.03
    ymin =  y_center*0.94
    ymax =  y_center*1.06
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('b')
    ax.set_ylabel('f')
    #plt.legend(elllist, labellist, loc=4, prop={'size':10})
    #plt.scatter(x_mean, y_mean)
    #plt.title( title )
    
    #pdf_name = pdfname
    #pdf=PdfPages(pdf_name)
    #pdf.savefig(fig)
    #pdf.close()
    #print "\n pdf file saved : ", pdf_name
    """



def FisherProjection( deriv, CovMatrix ):
    
    """ Projection for Fisher Matrix """
    inverseC = pinv(CovMatrix)
    #print np.allclose(CovMatrix, np.dot(CovMatrix, np.dot(inverseC, CovMatrix)))
    
    FisherMatrix = np.dot(np.dot(deriv, inverseC), np.transpose(deriv))
    
    for i in range(len(deriv)):
        for j in range(i, len(deriv)):
            FisherMatrix[j,i] = FisherMatrix[i,j]
    
    return FisherMatrix

def FisherProjection_Fishergiven( deriv, FisherM ):
    
    """ Projection for Fisher Matrix """
    
    FisherMatrix = np.dot(np.dot(deriv, FisherM), np.transpose(deriv))
    
    for i in range(len(deriv)):
        for j in range(i, len(deriv)):
            FisherMatrix[j,i] = FisherMatrix[i,j]
    
    
    return FisherMatrix

def avgBessel(l,k,rmin,rmax):
    
    """ 
    Calculate Averaged spherical Bessel function J_l (kr) in configuration space in each bin
    * fortranfunction module needed
    
    Parameters
    ----------
    l: order of spherical Bessel function (0,2,4)
    k: center value in each k bin
    rmin, rmax: minimum and maximum r values in each r bin
    
    """
    
    from numpy import vectorize, pi, cos, sin
    from fortranfunction import sici
    sici = vectorize(sici)
    
    if l == 0 :
        result = 4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin))/(k**3)
    elif l == 2 :
        result = 4. * pi * (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax) - 3*sici(k*rmin))/k**3
    else :
     
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * cos(k * rmax) +\
                  (- 105 * k/rmin + 2 * k**3 * rmin) * cos(k * rmin) +\
                  22 * k**2 * sin(k * rmax) - (105 * sin(k * rmax))/rmax**2 -\
                  22 * k**2 * sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                  15 * k**2 * (sici(k * rmax) - sici(k * rmin))) 
            
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * cos(k * rmax) +\
                                  (- 105 * k/rmin + 2 * k**3 * rmin) * cos(k * rmin) +\
                                  22 * k**2 * sin(k * rmax) - (105 * sin(k * rmax))/rmax**2 -\
                                  22 * k**2 * sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                                  15 * k**2 * (sici(k * rmax) - sici(k * rmin)))
    return result


class NoShell_covariance():


    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, N_y, b, f, s, nn, logscale = False):
        
        # const
        self.h= 1.0
        self.Vs= 5.0*10**9
        self.nn= nn # for shot noise. fiducial : 3x1e-04
        
        self.b= b
        self.f= f 
        self.s= s
        self.n3 = 2**4 + 1
        self.mulist, self.dmu = np.linspace(-1.,1., self.n3, retstep = True)
        
        # k scale range
        self.KMIN = KMIN
        self.KMAX = KMAX
        
        # r scale
        self.RMIN = RMIN
        self.RMAX = RMAX
        
        
        self.n = n #kN
        self.n2 = n2 #rN
        self.N_y = N_y #kN_y
    
        # evenly spaced bins -------------------------
        
        
        if logscale is False:
            # r bins setting
            self.rbin, dr = np.linspace(self.RMAX, self.RMIN, self.n2, retstep = True)
            self.dr = np.fabs(dr)
            self.rmin = np.delete(self.rbin,0)
            self.rmax = np.delete(self.rbin,-1)
            #self.rcenter = self.rmin + self.dr/2.
            self.rcenter = (3 * (self.rmax**3 + self.rmax**2 * self.rmin + self.rmax*self.rmin**2 + self.rmin**3))/(4 *(self.rmax**2 + self.rmax * self.rmin + self.rmin**2))
            
            # k bins setting
            #self.klist, self.dk = np.linspace(self.KMIN, self.KMAX, self.n + 1, retstep =True)
            
            self.kbin, self.dk = np.linspace(self.KMIN, self.KMAX, self.n, retstep = True)
            self.kmin = np.delete(self.kbin,-1)
            self.kmax = np.delete(self.kbin,0)
            #self.kcenter = self.kmin + self.dk/2.
            self.kcenter = (3 * (self.kmax**3 + self.kmax**2 * self.kmin + self.kmax*self.kmin**2 + self.kmin**3))/(4 *(self.kmax**2 + self.kmax * self.kmin + self.kmin**2))
            
            # k bin for xi integral setting
            #self.N_x = N_x
            #self.kbin_x, self.dk_x = np.linspace(0.001, 10, self.N_x, retstep = True)
            #self.kmin_x = np.delete(self.kbin_x,-1)
            #self.kmax_x = np.delete(self.kbin_x,0)
            #self.kcenter_x = self.kmin_x + self.dk_x/2.
            #self.kcenter_x = (3 * (self.kmax_x**3 + self.kmax_x**2 * self.kmin_x + self.kmax_x*self.kmin_x**2 + self.kmin_x**3))/(4 *(self.kmax_x**2 + self.kmax_x * self.kmin_x + self.kmin_x**2))
            
            
            self.kbin_y, self.dk_y = np.linspace(self.KMIN, self.KMAX, self.N_y, retstep = True)
            self.kmin_y = np.delete(self.kbin_y,-1)
            self.kmax_y = np.delete(self.kbin_y,0)
            #self.kcenter_y = self.kmin_y + self.dk_y/2.
            self.kcenter_y = (3 * (self.kmax_y**3 + self.kmax_y**2 * self.kmin_y + self.kmax_y*self.kmin_y**2 + self.kmin_y**3))/(4 *(self.kmax_y**2 + self.kmax_y * self.kmin_y + self.kmin_y**2))
        
        
        if logscale is True:
        
            self.rbin = np.logspace(np.log(self.RMAX),np.log(self.RMIN),self.n2, base = np.e)
            rbin = self.rbin
            self.rmin = np.delete(rbin,0)
            self.rmax = np.delete(rbin,-1)
            self.rcenter = np.array([ np.sqrt(rbin[i] * rbin[i+1]) for i in range(len(rbin)-1) ])
            self.dlnr = np.fabs(np.log(self.rbin[2]/self.rbin[3]))
            self.dr = np.fabs(self.rmax - self.rmin)
            
            # k bin for xi integral setting
            self.kbin = np.logspace(np.log10(KMIN),np.log10(KMAX), self.n, base=10)
            self.kcenter = np.array([(np.sqrt(self.kbin[i] * self.kbin[i+1])) for i in range(len(self.kbin)-1)])
            self.kmin = np.delete(self.kbin,-1)
            self.kmax = np.delete(self.kbin,0)
            self.dk = self.kmax - self.kmin
            self.dlnk = np.log(self.kbin[3]/self.kbin[2])
            
            self.kbin_y = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), self.N_y, base=10)
            self.kcenter_y = np.array([(np.sqrt(self.kbin_y[i] * self.kbin_y[i+1])) for i in range(len(self.kbin_y)-1)])
            self.kmin_y = np.delete(self.kbin_y,-1)
            self.kmax_y = np.delete(self.kbin_y,0)
            self.dk_y = self.kmax_y - self.kmin_y
            self.dlnk_y = np.log(self.kbin_y[3]/self.kbin_y[2])
        

    def compile_fortran_modules(self):
        """
        Generate python module from fortran subroutines of legendre and sine integral functions
        
        * f2py module needed
        * fortran file: fortranfunction.f90
        
        """
        
        def compile():
            print 'compiling fortran subroutines'
            import numpy.f2py.f2py2e as f2py2e
            import sys
            sys.argv +=  "-c -m fortranfunction fortranfunction.f90".split()
            f2py2e.main()
            sys.argv = [sys.argv[0]]
    
        def nocompile(): print 'skip fortran subroutine compiling'
        
        switch = {
            "y" : compile,
            "n" : nocompile }
        
        message = raw_input("first try? [y/n]")
        
        switch.get(message, nocompile)()

    

    def MatterPower(self, file ='matterpower_z_0.55.dat'):
        """
        Load power spectrum values from input file and generate sampling points
        
        Parmaeter
        ---------
        file: position and power spectrum values from camb
        
        """
        fo = open(file, 'r')
        position = fo.seek(0, 0)
        Pkl=np.array(np.loadtxt(fo))
        k=np.array(Pkl[:,0])
        P=np.array(Pkl[:,1])

        #power spectrum interpolation
        Pm = interp1d(k, P, kind= "linear")
        #self.Pmlist = Pm(self.kcenter)
        #self.RealPowerBand = Pm(self.kcenter)
        self.Pm_interp = Pm
        self.RealPowerBand = Pm(self.kcenter)
        self.RealPowerBand_y = Pm(self.kcenter_y)


    def multipole_P(self,l):
        """
        Calculate power spectrum multipoles up to quadrupole
        
        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """

        b = self.b
        f = self.f
        s = self.s
        
        
        kbin = self.kbin
        kcenter= self.kcenter_y
        mulist = self.mulist
        dmu = self.dmu
        PS = self.Pm_interp(kbin)
        
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kbin.size]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)
        
        kmatrix = kbin[matrix2]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0: Dmatrix = 1.
        R = (b + f * mumatrix**2)**2 * Dmatrix * Le_matrix
        Pmultipole = (2 * l + 1.)/2 * PS * romberg( R, dx=dmu, axis=0 )
        if l==0 : Pmultipole+= 1./self.nn
        
        Pmultipole_interp = interp1d(kbin, Pmultipole, kind= "linear")
        #self.Pmlist = Pm(self.kcenter)
        #self.RealPowerBand = Pm(self.kcenter)
        if l ==0 : self.Pmultipole0_interp = Pmultipole_interp
        elif l ==2 : self.Pmultipole2_interp = Pmultipole_interp
        elif l ==4 : self.Pmultipole4_interp = Pmultipole_interp
        else : raise ValueError('l should be 0, 2, 4')
            
        return Pmultipole_interp(kcenter)


    def multipole_P_band_all(self):

        from multiprocessing import Process, Queue
        def multipole_P_process(q, order, (l)):
            q.put((order, self.multipole_P(l)))
            #sys.stdout.write('.')
        
        inputs = (0,2,4)
        d_queue = Queue()
        d_processes = [Process(target=multipole_P_process, args=(d_queue, z[0], z[1])) for z in zip(range(3), inputs)]
        for p in d_processes:
            p.start()

        result = [d_queue.get() for p in d_processes]
    
        result.sort()
        result1 = [D[1] for D in result]

        self.multipole_bandpower0 = result1[0]
        self.multipole_bandpower2 = result1[1]
        self.multipole_bandpower4 = result1[2]
      
    
    def multipole_Xi(self,l):
        """
        Calculate power spectrum multipoles up to quadrupole
        
        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """
        import cmath
        I = cmath.sqrt(-1)
        

        kbin = self.kbin_y
        rcenter = self.rcenter

        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist

        #Pmlist = self.Pmlist
        #Pm = self.Pm_interp(kbin)
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn

        matrix1,matrix2 = np.mgrid[0:len(kbin),0:len(rcenter)]
        kmatrix = kbin[matrix1]
        
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rmatrix = rcenter[matrix2]
        Vir = 4./3 * np.fabs(rmax**3 - rmin**3)
        
        if l == 0 : Pm = self.Pmultipole0_interp(kbin) - 1./self.nn
        elif l == 2 : Pm = self.Pmultipole2_interp(kbin)
        elif l == 4 : Pm = self.Pmultipole4_interp(kbin)
        else : raise ValueError('l should be 0, 2, 4')
            
        Pmatrix = Pm[matrix1]
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
        
        #AvgBessel = np.array([ avgBessel(l, k ,rmin, rmax) for k in kcenter ])/Vir
        #AvgBessel = avgBessel(l, kmatrix, rminmatrix, rmaxmatrix )/Vir
        AvgBessel = sbess(l, kmatrix * rmatrix)
        multipole_xi = np.real(I**l) * simpson(kmatrix**2 * Pmatrix * AvgBessel/(2*np.pi**2), kmatrix, axis=0)#/Vir

        return multipole_xi
    
    
           
            
    def covariance_PP(self, l1, l2):

        #from scipy.integrate import simps, romb
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        kbin = self.kbin_y
        kcenter = self.kcenter_y
        #skbin = self.skbin
        mulist = self.mulist
        dk = self.kmax_y - self.kmin_y
        #dlnk = self.dlnk
        #sdlnk = self.sdlnk
        #sdk = self.skmax - self.skmin
        PS = self.RealPowerBand_y
        
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        dmu = self.dmu
        
        # FirstTerm + SecondTerm
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kcenter.size]
        mumatrix = self.mulist[matrix1]
        
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        Vi = 4./3 * pi * ( self.kmax_y**3 - self.kmin_y**3 ) #4*np.pi*kcenter**2
        
        Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) * (2*pi)**3 /Vs
        
        kmatrix = kcenter[matrix2]
        Pmmatrix = PS[matrix2]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0 : Dmatrix = 1.
        R = (b + f * mumatrix**2)**2 * Dmatrix
        
        Rintegral3 = Const_alpha * PS**2 * romberg( R**2 * Le_matrix1 * Le_matrix2, dx=dmu, axis=0 )/Vi
        Rintegral2 = Const_alpha * 2./nn * PS * romberg( R * Le_matrix1 * Le_matrix2, dx=dmu, axis=0 )/Vi
     

        FirstSecond = Rintegral3 + Rintegral2

        
        # LastTerm
        if l1 == l2:
            LastTerm = (2*l1 + 1.) * 2. * (2 * pi)**3/Vs/nn**2 /Vi
        else:
            LastTerm = np.zeros(FirstSecond.shape)
        
        Total = FirstSecond + LastTerm
        #Total = LastTerm
        covariance_mutipole_PP = np.zeros((kcenter.size,kcenter.size))
        np.fill_diagonal(covariance_mutipole_PP,Total)

        #print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        
        #sys.stdout.write('.')
        
        return covariance_mutipole_PP


    def RSDband_covariance_PP_all(self):
        
        
        from multiprocessing import Process, Queue
        def RSDband_covariance_PP_process(q, order, (l1, l2)):
            q.put((order, self.covariance_PP(l1, l2)))
            #sys.stdout.write('.')
        
        inputs = ((0,0), (0,2), (0,4), (2,2), (2,4), (4,4))
        d_queue = Queue()
        d_processes = [Process(target=RSDband_covariance_PP_process, args=(d_queue, z[0], z[1])) for z in zip(range(6), inputs)]
        for p in d_processes:
            p.start()
    
        result = []
        percent = 0.0
        for d in d_processes:
            result.append(d_queue.get())
            percent += + 1./len( d_processes ) * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()
        
        #result = [d_queue.get() for p in d_processes]
        
        result.sort()
        result1 = [D[1] for D in result]
        

        
        self.covariance_PP00 = result1[0]
        self.covariance_PP02 = result1[1]
        self.covariance_PP04 = result1[2]
        self.covariance_PP22 = result1[3]
        self.covariance_PP24 = result1[4]
        self.covariance_PP44 = result1[5]

    
    def derivative_Xi(self, l):
        """
        Calculate derivatives dXi/dP up to mode l=4
        dxi_l / dp_li = i^l /int(k^2 ShellavgBessel(kr) / 2pi^2), from kmin to kmax
        
        Parameters
        ----------
        l: mode 0,2,4
        
        """

        from fortranfunction import sbess
        import cmath
        I = cmath.sqrt(-1)

        #kbin = self.klist
        kcenter = self.kcenter_y
        dk = self.dk_y
        #sdlnk = self.sdlnk
        mulist = self.mulist
        #dlnk = self.dlnk
        rcenter = self.rcenter
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
        kmin = self.kmin_y
        kmax = self.kmax_y
        
        matrix1, matrix2 = np.mgrid[ 0: kcenter.size, 0: rcenter.size ]
        #matrix3, matrix4 = np.mgrid[ 0: skbin.size 0: rcenter.size]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rmatrix = rcenter[matrix2]
        kmatrix = kcenter[matrix1]
        kminmatrix = kmin[matrix1]
        kmaxmatrix = kmax[matrix1]
        #Vir = 4 * pi * rcenter**2 * dr + 1./3 * pi * dr**3
        #Vir = 4./3 * pi * np.fabs(rmaxmatrix**3 - rminmatrix**3)
        #AvgBesselmatrix = avgBessel(l, kmatrix ,rminmatrix,rmaxmatrix)/Vir
        #intmatrix = np.real(I**l) * kmatrix**2/(2*pi**2) * AvgBesselmatrix

        #Inds = np.digitize( skbin, kbin )
        

        #AvgBessel = avgBessel(l, kmatrix ,rminmatrix, rmaxmatrix)/Vir
        #sbess = np.vectorize(sbess)
        #Bessel = sbess(l, kmatrix * rmatrix)
        #derivative_Xi_band = np.real(I**l) * kmatrix**2/(2*pi**2) * dk * Bessel
        derivative_Xi_band = np.real(I**l) * avgBessel(l,rmatrix,kminmatrix,kmaxmatrix) / (2 * pi)**3
        #resultlist.append(integral)

        """
        resultlist=[]
        for j in range(rcenter.size):
            AvgBessel = avgBessel(l, kcenter ,rmin[j],rmax[j])/Vir[j]
            integral = np.real(I**l) * kcenter**2/(2*pi**2) * dk * AvgBessel
            resultlist.append(integral)
        
        derivative_Xi_band = np.array(resultlist).reshape( rcenter.size, kcenter.size ).T
        """
        
        #sys.stdout.write('.')
        return derivative_Xi_band

            
    def derivative_Xi_band_all(self):
        """
        Do pararrel process for function derivative_Xi_band() and calculate results of all modes(l=0,2,4) at once
        
        * parrarel python needed
        
        """
        
        from multiprocessing import Process, Queue
        def derivative_Xi_band_process(q, order, (l)):
            q.put((order, self.derivative_Xi(l)))
            #sys.stdout.write('.')

        inputs = ((0),(2),(4))
        d_queue = Queue()
        d_processes = [Process(target=derivative_Xi_band_process, args=(d_queue, z[0], z[1])) for z in zip(range(3), inputs)]
        for p in d_processes:
            p.start()

        #result = [d_queue.get() for p in d_processes]
    
        result = []
        percent = 0.0
        for d in d_processes:
            result.append(d_queue.get())
            percent += + 1./len( d_processes ) * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()
    
    
        result.sort()
        result1 = [D[1] for D in result]
        
        self.dxip0 = result1[0]
        self.dxip2 = result1[1]
        self.dxip4 = result1[2]



    def derivative_bfs(self,l):
    
    
        import cmath
        I = cmath.sqrt(-1)
        
        kbin = self.kbin
        rbin = self.rbin
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk
        dr = self.dr
        dmu = self.dmu
        #Pmlist = self.Pmlist
        Pm = self.Pm_interp(kbin)
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
    
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(kbin)]
        mumatrix = mulist[matrix1] # mu matrix (axis 0)
        kmatrix = kbin[matrix2] # k matrix (axis 1)
        Le_matrix = Ll(l, mumatrix)
        #Pmmatrix = Pm[matrix2]
        const =  np.real(I**l) * (2 * l + 1)/2. / (2 * np.pi**2)
        Dmatrix = np.exp(-kmatrix**2 * mumatrix**2 * self.s**2)
        if self.s == 0 : Dmatrix = 1.
        db =  const* (b + f* mumatrix**2)*2 * Le_matrix *Dmatrix
        df =  const* (b + f*mumatrix**2)*2 * mumatrix**2 * Le_matrix * Dmatrix
        ds =  const* (b + f*mumatrix**2)**2 * (-kmatrix**2 * mumatrix**2) * Le_matrix * Dmatrix
        # mu integration
        muintb = romberg(db ,dx=dmu, axis=0) # return 1d array along k-axis
        muintf = romberg(df ,dx=dmu, axis=0) # return 1d array along k-axis
        muints = romberg(ds ,dx=dmu, axis=0) # return 1d array along k-axis
    
        matrix1,matrix2 = np.mgrid[0:len(kbin),0:len(rcenter)]
        kmatrix = kbin[matrix1]
        Pmmatrix = Pm[matrix1]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rmatrix = rcenter[matrix2]
        Vir = 4./3 * np.fabs(rmax**3 - rmin**3)
        intb = muintb[matrix1]
        intf = muintf[matrix1]
        ints = muints[matrix1]
        
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
        
        #AvgBessel = np.array([ avgBessel(l, k ,rmin, rmax) for k in kcenter ])/Vir
        #AvgBessel = avgBessel(l, kmatrix, rminmatrix, rmaxmatrix )
        AvgBessel = sbess(l, kmatrix * rmatrix)
        Total_Integb = simpson(kmatrix**2 * Pmmatrix * intb * AvgBessel, kmatrix, axis=0)#/Vir
        Total_Integf = simpson(kmatrix**2 * Pmmatrix * intf * AvgBessel, kmatrix, axis=0)#/Vir
        Total_Integs = simpson(kmatrix**2 * Pmmatrix * ints * AvgBessel, kmatrix, axis=0)#/Vir

        return Total_Integb, Total_Integf, Total_Integs
    
    def derivative_bfs_all(self):
    
        self.dxib0, self.dxif0, self.dxis0 = self.derivative_bfs(0)
        self.dxib2, self.dxif2, self.dxis2 = self.derivative_bfs(2)
        self.dxib4, self.dxif4, self.dxis4 = self.derivative_bfs(4)
    
    

    def covariance_Xi_all(self):
        """
        Calculate Xi covariance matrices for all possible cases at once (9 sub matrices)
        
        * need multiprocessing module
        
        """
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)    

        #kcenter = self.kcenter
        kbin = self.kbin
        rbin = self.rbin
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dmu = self.dmu
        #dk = self.dk
        dr = self.dr
        Pm = self.Pm_interp(kbin)
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        
    
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(kbin)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = kbin[matrix2] # k matrix (axis 1)
        Le_matrix0 = Ll(0,mulistmatrix)
        Le_matrix2 = Ll(2,mulistmatrix)
        Le_matrix4 = Ll(4,mulistmatrix)
    
        
        Dmatrix = np.exp(-klistmatrix**2 * mulistmatrix**2 * self.s**2)
        if self.s == 0 : Dmatrix = 1.
        R = (b + f * mulistmatrix**2)**2 * Dmatrix

        from multiprocessing import Process, Queue
        
        """print 'Rintegral' """
        def Rintegral(q, order, (l1, l2, Le1, Le2)):
            
            #import covariance_class2
            from numpy import pi, real
            #from scipy.integrate import simps
            import cmath
            
            I = cmath.sqrt(-1)
            const_gamma = real(I**(l1+l2)) * 2.* (2*l1+1)*(2*l2+1) /(2*pi)**2 /Vs
            muint3 = romberg(R**2 * Le1 * Le2, dx=dmu, axis=0 )
            muint2 = romberg(R * Le1 * Le2, dx=dmu, axis=0 )
            result = const_gamma * (muint3 * Pm**2 + muint2 * Pm * 2./nn)
            #sys.stdout.write('.')
            
            q.put((order,result))
        
        inputs = (( 0, 0, Le_matrix0, Le_matrix0),( 0, 2, Le_matrix0, Le_matrix2),(0, 4,Le_matrix0, Le_matrix4),(2, 2, Le_matrix2, Le_matrix2),(2, 4, Le_matrix2, Le_matrix4),(4, 4, Le_matrix4, Le_matrix4))
        
        R_queue = Queue()
        R_processes = [Process(target=Rintegral, args=(R_queue, z[0], z[1])) for z in zip(range(6), inputs)]
        for p in R_processes:
            p.start()
        
        #Rintegrals = [R_queue.get() for p in R_processes]

        Rintegrals = []
        percent = 0.0
        for d in R_processes:
            Rintegrals.append(R_queue.get())
            percent += + 1./len( R_processes )/3 * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()

        Rintegrals.sort()
        Rintegrallist = [R[1] for R in Rintegrals]
        
        Rintegral00 = Rintegrallist[0] # 1D
        Rintegral02 = Rintegrallist[1]
        Rintegral04 = Rintegrallist[2]
        Rintegral22 = Rintegrallist[3]
        Rintegral24 = Rintegrallist[4]
        Rintegral44 = Rintegrallist[5]
    
        matrix4,matrix5 = np.mgrid[0:len(rcenter),0:len(rcenter)]
        rbinmatrix1 = rcenter[matrix4] # vertical
        rbinmatrix2 = rcenter[matrix5] # horizontal
        rminmatrix = rmin[matrix4] # vertical
        rminmatrix2 = rmin[matrix5] # horizontal
        rmaxmatrix = rmax[matrix4] # vertical
        rmaxmatrix2 = rmax[matrix5] # horizontal
        dr1 = np.fabs(rmaxmatrix - rminmatrix)
        dr2 = np.fabs(rmaxmatrix2 - rminmatrix2)
        Vir1 = 4./3 * pi * np.fabs(rminmatrix**3 - rmaxmatrix**3)
        Vir2 = 4./3 * pi * np.fabs(rminmatrix2**3 - rmaxmatrix2**3)
        Vi = 4./3 * pi * np.fabs(rmin**3 - rmax**3)
        """
        def AvgBessel_q(q, order, (l, kbin, rmin, rmax)):
            Avg = [avgBessel(l,k,rmin,rmax) for k in kbin] #2D (kxr)
            #sys.stdout.write('.')
            q.put((order,Avg))
        #inputs_bessel = [(0, kbin,rmin,rmax),(2, kbin,rmin,rmax), (4, kbin,rmin,rmax) ]
        """
        def AvgBessel_q(q, order, (l, kbin, rcenter)):
            Avg = [sbess(l,k * rcenter) for k in kbin] #2D (kxr)
            sys.stdout.write('.')
            q.put((order,Avg))
        inputs_bessel = [(0.0, kbin,rcenter),(2.0, kbin,rcenter), (4.0, kbin,rcenter) ]
        
        B_queue = Queue()
        B_processes = [Process(target=AvgBessel_q, args=(B_queue,z[0], z[1])) for z in zip(range(3), inputs_bessel)]
        
        for pB in B_processes:
            pB.start()
        
        #Bessels = [B_queue.get() for pB in B_processes]
        
        Bessels = []
        for pB in B_processes:
            Bessels.append(B_queue.get())
            percent += + 1./len( B_processes )/3 * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()
        
        
        Bessels.sort()
        Bessel_list = [ B[1] for B in Bessels] #2D bessel, (kxr)


        avgBesselmatrix0 = np.array(Bessel_list[0]) #2D, (kxr)
        avgBesselmatrix2 = np.array(Bessel_list[1])
        avgBesselmatrix4 = np.array(Bessel_list[2])

        matrix1, matrix2 = np.mgrid[0:len(kbin), 0:len(rcenter)]
        Volume_double = Vir1 * Vir2
        kmatrix = kbin[matrix1]
        #try:
        #    if dk.size == 1: pass
        #    else : dk = dk[matrix1]
        #except (AttributeError): pass
        
        #Vik = 4./3 * pi * np.fabs(self.kmax**3 - self.kmin**3)
        #Vik = Vik[matrix1]
        
        
        
        def FirstSecond(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
        
            Rint_result = result[matrix1] # 2D
            #sdk = self.sdk[matrix1]
            
            relist = []
            for i in range(len(rcenter)/2):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                re = simpson(Rint_result * avgBmatrix * avgBessel2 * kmatrix**2, kmatrix, axis=0)
                #re = np.sum(Rint_result * Vik/(4*pi) * avgBmatrix * avgBessel2, axis=0)
                #re = np.sum(Rint_result * dk * kmatrix**2 * avgBmatrix * avgBessel2, axis=0)
                relist.append(re)
            FirstTerm = np.array(relist)#  / Volume_double[0:len(rcenter)/2,:] #2D
            
            LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array    
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[0:len(rcenter)/2,:]
            else : LastTerm = LastTermmatrix[0:len(rcenter)/2,:]
            
        
            re = FirstTerm+LastTerm
            #re = LastTerm
            queue.put((order,re))
            #sys.stdout.write('.')

        def FirstSecond2(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
    
            Rint_result = result[matrix1] # 2D
            #sdk = self.sdk[matrix1]
            
            relist = []
            for i in range(len(rcenter)/2, len(rcenter)):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                
                re = simpson(Rint_result * avgBmatrix * avgBessel2 * kmatrix**2,kmatrix, axis=0)
                #re = np.sum(Rint_result * Vik/(4*pi)* avgBmatrix * avgBessel2, axis=0)
                #re = np.sum(Rint_result * dk * kmatrix**2 * avgBmatrix * avgBessel2, axis=0)
                relist.append(re)
            FirstTerm = np.array(relist) #/ Volume_double[len(rcenter)/2:len(rcenter),:] #2D
            
            LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[len(rcenter)/2:,:]
            else : LastTerm = LastTermmatrix[len(rcenter)/2:,:]
            
            re = FirstTerm+LastTerm
            #re = LastTerm
            queue.put((order,re))
            #sys.stdout.write('.')
        
        F_inputs = (( 0, 0, Rintegral00, avgBesselmatrix0, avgBesselmatrix0),( 0, 2, Rintegral02,  avgBesselmatrix0, avgBesselmatrix2),(0, 4, Rintegral04, avgBesselmatrix0, avgBesselmatrix4 ),(2, 2, Rintegral22, avgBesselmatrix2, avgBesselmatrix2 ),(2, 4, Rintegral24, avgBesselmatrix2, avgBesselmatrix4 ),(4, 4, Rintegral44, avgBesselmatrix4, avgBesselmatrix4))
        
        F_queue = Queue()
        F_processes1 = [Process(target=FirstSecond, args=(F_queue, z[0], z[1])) for z in zip(range(6),F_inputs)]
        F_processes2 = [Process(target=FirstSecond2, args=(F_queue, z[0], z[1])) for z in zip(range(6,12),F_inputs)]
        F_processes = F_processes1 + F_processes2
        
        
        for pF in F_processes:
            pF.start()

        Ts = []

        for pF in F_processes:
            Ts.append(F_queue.get())
            percent += + 1./len( F_processes )/3 * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()

        #Ts = [F_queue.get() for pF in F_processes]
        Ts.sort()
        Total = [T[1] for T in Ts]

        self.covariance00 = np.vstack((Total[0], Total[6]))
        self.covariance02 = np.vstack((Total[1], Total[7]))
        self.covariance04 = np.vstack((Total[2], Total[8]))
        self.covariance22 = np.vstack((Total[3], Total[9]))
        self.covariance24 = np.vstack((Total[4], Total[10]))
        self.covariance44 = np.vstack((Total[5], Total[11]))



    def covariance_PXi(self, l1, l2):
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
                
        import cmath
        I = cmath.sqrt(-1)
        
        klist = self.kbin_y
        kcenter = self.kcenter_y
        kmin = self.kmin_y
        kmax = self.kmax_y
        #skbin = self.skbin
        #skcenter =self.skcenter
        #sdlnk = self.sdlnk
        rlist = self.rbin
        rcenter = self.rcenter
        #dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        
        matrix1, matrix2 = np.mgrid[ 0:kcenter.size, 0: rcenter.size]
        kmatrix = kcenter[matrix1]
        rmatrix = rcenter[matrix2]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
     
        Besselmatrix = sbess(l2, kmatrix * rmatrix)
        #Vir = 4/3. * pi * ( rmaxmatrix**3 - rminmatrix**3)
        #Besselmatrix = avgBessel(l2,kmatrix,rminmatrix,rmaxmatrix)/Vir
        
        Vi = 4./3 * pi * ( self.kmax_y**3 - self.kmin_y**3 ) #4*np.pi*kcenter**2
        
        if l1 == 0 :
            if l2 == 0 : Cll = self.covariance_PP00.diagonal() * Vi
            elif l2 == 2 : Cll = self.covariance_PP02.diagonal() * Vi
            else : Cll = self.covariance_PP04.diagonal() * Vi
        elif l1 == 2 :
            if l2 == 0 : Cll = self.covariance_PP02.diagonal() * Vi
            elif l2 == 2 : Cll = self.covariance_PP22.diagonal() * Vi
            else : Cll = self.covariance_PP24.diagonal() * Vi
        elif l1 == 4 :
            if l2 == 0 : Cll = self.covariance_PP04.diagonal() * Vi
            elif l2 == 2 : Cll = self.covariance_PP24.diagonal() * Vi
            else : Cll = self.covariance_PP44.diagonal()* Vi
        else : raise ValueError('l should be 0,2,4')
        Cll = np.real(I**(l2)) * Cll * kcenter**2 /(2*np.pi**2) #Vi/(2*np.pi)**3
        Cpxill = Cll[matrix1] * Besselmatrix 
        return Cpxill
        
        
    def _covariance_PXi( self, l1, l2 ):


        import numpy as np
        from numpy import zeros, sqrt, pi, sin, cos, exp
        from numpy.linalg import inv
        from numpy import vectorize
        #from scipy.integrate import simps, simps
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
        
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.kbin_y
        kcenter = self.kcenter_y
        kmin = self.kmin_y
        kmax = self.kmax_y
        #skbin = self.skbin
        #skcenter =self.skcenter
        #sdlnk = self.sdlnk
        rlist = self.rbin
        rcenter = self.rcenter
        #dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        #dk = self.dk
        #dlnk = self.dlnk
        dr = self.dr
        #Pmlist = self.Pmlist
        Pmlist = self.RealPowerBand_y
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        dmu = self.dmu

        Const_beta = np.real(I**(l2)) * (2*l1 + 1.) * (2*l2 + 1.)/Vs
        matrix1, matrix2 = np.mgrid[0:len(mulist), 0: len(kcenter)]
        kmatrix = kcenter[matrix2]
        mumatrix = mulist[matrix1]
        Le_matrix1 = Ll(l1,mulist)[matrix1]
        Le_matrix2 = Ll(l2,mulist)[matrix1]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0 : Dmatrix = 1.
        R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
        Rintegral3 = romberg( R**2 * Le_matrix1 * Le_matrix2, dx = dmu, axis= 0 )
        Rintegral2 = romberg( R * Le_matrix1 * Le_matrix2, dx = dmu, axis= 0 )
        
        matrix1, matrix2 = np.mgrid[ 0:kcenter.size, 0: rcenter.size]
        kmatrix = kcenter[matrix1]
        kminmatrix = kmin[matrix1]
        kmaxmatrix = kmax[matrix1]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rmatrix = rcenter[matrix2]
        Vir = 4/3. * pi * ( rmaxmatrix**3 - rminmatrix**3)
        #Vik = 4/3. * pi * ( kmaxmatrix**3 - kminmatrix**3)
        AvgBesselmatrix = avgBessel(l2,kmatrix,rminmatrix,rmaxmatrix)/Vir
        #AvgBesselmatrix = avgBessel(l2,rmatrix,kminmatrix,kmaxmatrix) #/Vik
        #AvgBesselmatrix = sbess(l2, kmatrix * rmatrix)
        Rintegral3 = Rintegral3[matrix1]
        Rintegral2 = Rintegral2[matrix1]
        Pmmatrix = Pmlist[matrix1]
        
        FirstTerm = Const_beta * np.array((Pmmatrix**2 * Rintegral3 + Pmmatrix * 2/nn * Rintegral2 ) * AvgBesselmatrix).reshape((len(kcenter), len(rcenter)))
        #SecondTerm = Const_beta * np.array(Pmmatrix * 2/nn * Rintegral2 * AvgBesselmatrix).reshape((len(kcenter), len(rcenter)))
        
        if l1 == l2 :
            LastTerm = np.real(I**(l2)) * (2*l2 + 1.)*2 /Vs/nn**2 * AvgBesselmatrix.reshape((len(kcenter), len(rcenter)))
        else : LastTerm = np.zeros((len(kcenter), len(rcenter)))
        
        covariance_multipole_PXi = FirstTerm + LastTerm
        #covariance_multipole_PXi = LastTerm
        
        #print 'covariance_PXi {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_multipole_PXi


    def covariance_PXi_All(self):
        
        #import pp, sys, time
        
        from multiprocessing import Process, Queue
        
        def PXi_process(q, order, (l1, l2)):
            re = self.covariance_PXi(l1, l2)
            q.put((order, re))
        
        inputs = ((0, 0),(0, 2),(0, 4),(2, 0),(2, 2),(2, 4),(4, 0),(4, 2),(4, 4))
        q = Queue()
        Processes = [Process(target = PXi_process, args=(q, z[0], z[1])) for z in zip(range(9), inputs)]
        for p in Processes: p.start()
        result = [q.get() for p in Processes]
        result.sort()
        result1 = [result[i][1] for i in range(len(result)) ]
        
        self.covariance_PXi00 = result1[0]
        self.covariance_PXi02 = result1[1]
        self.covariance_PXi04 = result1[2]
        self.covariance_PXi20 = result1[3]
        self.covariance_PXi22 = result1[4]
        self.covariance_PXi24 = result1[5]
        self.covariance_PXi40 = result1[6]
        self.covariance_PXi42 = result1[7]
        self.covariance_PXi44 = result1[8]



    def derivative_P_bfs(self,l):
        
        """ dP_l/dq """
  
        b = self.b
        f = self.f
        s = self.s
        
        klist = self.kbin_y
        kcenter = self.kcenter_y
        #skbin = self.skbin
        #skcenter = self.skcenter
        #dk = self.dk
        dmu = self.dmu
        mulist = self.mulist
        #dlnk = self.dlnk
        #Pmlist = self.Pmlist
        Pm = self.RealPowerBand_y
        
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kcenter.size]
        
        kmatrix = kcenter[matrix2]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mulist)[matrix1]
        
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        Vi = 4./3 * pi * (self.kmax_y**3 - self.kmin_y**3)
        Pmmatrix = Pm[matrix2]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0:Dmatrix = 1.
        Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
        Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
        Rs = (- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix# * Le_matrix
        Rintb = (2 * l + 1.)/2 * romberg( Pmmatrix * Rb * Le_matrix, dx=dmu, axis=0 )
        Rintf = (2 * l + 1.)/2 * romberg( Pmmatrix * Rf * Le_matrix, dx=dmu, axis=0 )
        Rints = (2 * l + 1.)/2 * romberg( Pmmatrix * Rs * Le_matrix, dx=dmu, axis=0 )

        return Rintb, Rintf, Rints
    
    
    def derivative_P_bfs_all(self):
    
        self.dPb0, self.dPf0, self.dPs0 = self.derivative_P_bfs(0)
        self.dPb2, self.dPf2, self.dPs2 = self.derivative_P_bfs(2)
        self.dPb4, self.dPf4, self.dPs4 = self.derivative_P_bfs(4)


        
        
class NoShell_covariance_MCMC(NoShell_covariance):

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, N_y, b, f, s, nn, logscale = False):
        NoShell_covariance.__init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, N_y, b, f, s, nn, logscale = logscale)
        
    def init_like_class(self, datav_filename, fisher_filename, mask_filename):
        
        print ' Read data...'
        print ' datav  : ', datav_filename
        print ' fisher : ', fisher_filename
        print ' mask   : ', mask_filename
        
        self.datav_fid = np.genfromtxt(datav_filename)
        self.fisher = np.genfromtxt(fisher_filename)
        mask = np.genfromtxt(mask_filename)
        self.mask = np.array(mask, dtype=bool)
        
        print ' N data point after masking:', self.datav_fid[self.mask].size

        