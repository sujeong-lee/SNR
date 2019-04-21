import numpy as np
from numpy import zeros, sqrt, pi, sin, cos, exp
from numpy.linalg import pinv as inv
from numpy import vectorize
from scipy.interpolate import interp1d
#from scipy.integrate import simps
import sys
import matplotlib.pyplot as plt
from scipy_integrate import *



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
    Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)


    from numpy import vectorize, pi, cos, sin
    from fortranfunction import sici
    sici = vectorize(sici)
    
    if l == 0 :
        result = 4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin))/(k**3)
    elif l == 2 :
        result = 4. * pi * (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax) - 3*sici(k*rmin))/k**3
    elif l == 4:
     
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
        
    else : raise ValueError('only support l = 0,2,4')


    return result/Vir



def log_interp(x, y):
    import scipy
    from numpy import log, exp
    s = scipy.interpolate.interp1d(log(x), log(y))
    x0 = x[0]
    y0 = y[0]
    x1 = x[-1]
    y1 = y[-1]

    def interpolator(xi):
        w1 = xi == 0
        w2 = (xi > 0) & (xi <= x0)
        w3 = xi >= x1
        w4 = ~ (w1 | w2 | w3)

        y = np.zeros_like(xi)
        y[w2] = y0 * (x0 / xi[w2])
        y[w3] = y1 * (x1 / xi[w3])**3
        y[w4] = exp(s(log(xi[w4])))
        return y
    return interpolator


class class_covariance():


    def __init__(self, KMIN=1e-04, KMAX=50, RMIN=0.1, RMAX=180, n=20000, n_y = 200, n2=200, b=2, f=0.74, s=3.5, nn=3.0e-04, kscale = 'log', rscale='lin'):

        """
        class_covariance : class
        should be initialized first 


        Input params
        ------------
        nn : shot noise

        b : galaxy bias
        f : growth rate dlnD/dlna
        s : velocity dispersion in FoG term

        KMIN / KMAX : k range for Fourier transform
        RMIN / RMAX : r scale 

        n : number of k sampling points for Fourier transform
        n2 : number of r bin

        kscale : k bin spacing. 'log' or 'lin' 
        rscale : r bin spacing. 'log' or 'lin'



        Internal params 
        ----------------
        h : h0
        Vs : survey volume
        n3 : number of mu(cos(theta)) bin
        mPk_file : matter power spectrum file

        """

        # const
        self.h= 1.0
        self.Vs= 5.0*10**9
        self.nn= nn # for shot noise. fiducial : 3x1e-04
        
        self.b= b
        self.f= f 
        self.s= s
        self.n3 = 2**8+1
        self.mulist, self.dmu = np.linspace(-1.,1., self.n3, retstep = True)
        
        # k scale range
        self.KMIN = KMIN
        self.KMAX = KMAX
        
        # r scale
        self.RMIN = RMIN
        self.RMAX = RMAX
        
        
        self.n = n #kN
        self.n2 = n2 #rN
        self.N_y = n_y #kN_y
    
        self.mPk_file = 'matterpower_z_0.55.dat'
        
        # k spacing for Fourier transform
        self.kbin = np.logspace(np.log10(KMIN),np.log10(KMAX), self.n, base=10)
        self.dlnk = np.log(self.kbin[3]/self.kbin[2])        
        self.kcenter = np.array([(np.sqrt(self.kbin[i] * self.kbin[i+1])) for i in range(len(self.kbin)-1)])        

        
        """
        if kscale is 'lin':
            self.kbin_y, self.dk_y = np.linspace(self.KMIN, self.KMAX, self.N_y, retstep = True)
            self.kmin_y = np.delete(self.kbin_y,-1)
            self.kmax_y = np.delete(self.kbin_y,0)
            self.kcenter_y = self.kmin_y + self.dk_y/2.

        elif kscale is 'log' : 
            self.kbin_y = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), self.N_y, base=10)
            self.kcenter_y = np.array([(np.sqrt(self.kbin_y[i] * self.kbin_y[i+1])) for i in range(len(self.kbin_y)-1)])
            self.kmin_y = np.delete(self.kbin_y,-1)
            self.kmax_y = np.delete(self.kbin_y,0)
            self.dk_y = self.kmax_y - self.kmin_y
            self.dlnk_y = np.log(self.kbin_y[3]/self.kbin_y[2])
            #self.kcenter = (3 * (self.kmax**3 + self.kmax**2 * self.kmin + self.kmax*self.kmin**2 + self.kmin**3))/(4 *(self.kmax**2 + self.kmax * self.kmin + self.kmin**2))
        """    
        
        if rscale is 'lin':
            # r bins setting
            self.rbin, dr = np.linspace(self.RMAX, self.RMIN, self.n2, retstep = True)
            self.dr = np.fabs(dr)
            self.rmin = np.delete(self.rbin,0)
            self.rmax = np.delete(self.rbin,-1)
            self.rcenter = self.rmin + self.dr/2.
            #self.rcenter = (3 * (self.rmax**3 + self.rmax**2 * self.rmin + self.rmax*self.rmin**2 + self.rmin**3))/(4 *(self.rmax**2 + self.rmax * self.rmin + self.rmin**2))
            
        elif rscale is 'log' :
            self.rbin = np.logspace(np.log(self.RMAX),np.log(self.RMIN),self.n2, base = np.e)
            rbin = self.rbin
            self.rmin = np.delete(rbin,0)
            self.rmax = np.delete(rbin,-1)
            self.rcenter = np.array([ np.sqrt(rbin[i] * rbin[i+1]) for i in range(len(rbin)-1) ])
            self.dlnr = np.fabs(np.log(self.rbin[2]/self.rbin[3]))
            self.dr = np.fabs(self.rmax - self.rmin)
        

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

    

    def MatterPower(self, file = None):
        """
        Load matter power spectrum values from input file and do log-interpolattion
        
        Parmaeter
        ---------
        file: txt file that consists of 2 columns k and Pk
        
        """
        if file is None : file = self.mPk_file
        fo = open(file, 'r')
        position = fo.seek(0, 0)
        Pkl=np.array(np.loadtxt(fo))
        k=np.array(Pkl[:,0])
        P=np.array(Pkl[:,1])

        #power spectrum interpolation
        #Pm = interp1d(k, P, kind= "linear")
        Pm = log_interp(k, P)
        #self.Pmlist = Pm(self.kcenter)
        #self.RealPowerBand = Pm(self.kcenter)
        self.Pm_interp = Pm
        #self.RealPowerBand = Pm(self.kcenter)
        #self.RealPowerBand_y = Pm(self.kcenter_y)

    

    def multipole_P(self,l):
        """
        Calculate power spectrum multipoles
        
        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """
        try: self.Pm_interp(1.0)
        except : self.MatterPower()
            
        b = self.b
        f = self.f
        s = self.s
        if self.nn == 0 : overnn = 0
        else : overnn = 1./self.nn
        
        kbin = self.kbin
        #kcenter= self.kcenter_y
        mulist = self.mulist
        dmu = self.dmu
        PS = self.Pm_interp(kbin)
        
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kbin.size]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)
        
        kmatrix = kbin[matrix2]
        Dmatrix = np.exp(- 1.*kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0: Dmatrix = 1.
        R = (b + f * mumatrix**2)**2 * Dmatrix * Le_matrix
        Pmultipole = (2 * l + 1.)/2. * PS * romberg( R, dx=dmu, axis=0 )
        if l==0 : Pmultipole+= overnn
        
        Pmultipole_interp = log_interp(kbin, Pmultipole)
        #self.Pmlist = Pm(self.kcenter)
        #self.RealPowerBand = Pm(self.kcenter)
        if l ==0 : self.Pmultipole0_interp = Pmultipole_interp
        elif l ==2 : self.Pmultipole2_interp = Pmultipole_interp
        elif l ==4 : self.Pmultipole4_interp = Pmultipole_interp
        else : raise ValueError('l should be 0, 2, 4')
            
        #return Pmultipole_interp(kcenter)
        return Pmultipole_interp(kbin)


    #def multipole_P_band_all(self):
    #    self.multipole_bandpower0 = self.multipole_P(0)
    #    self.multipole_bandpower2 = self.multipole_P(2)
    #    self.multipole_bandpower4 = self.multipole_P(4)      
        
    
    def multipole_Xi(self,l):
        """
        Calculate xi multipoles up to quadrupole
        
        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """
        import cmath
        I = cmath.sqrt(-1)
        if self.nn == 0 : overnn = 0
        else : overnn = 1./self.nn

            
        kbin = self.kbin
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
        #Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)
        
        try : self.Pmultipole0_interp
        except : self.multipole_P_band_all()
            
        if l == 0 : 
            try : Pm = self.Pmultipole0_interp(kbin) - overnn
            except (ZeroDivisionError): Pm = self.Pmultipole0_interp(kbin)
        elif l == 2 : Pm = self.Pmultipole2_interp(kbin)
        elif l == 4 : Pm = self.Pmultipole4_interp(kbin)
        else : raise ValueError('l should be 0, 2, 4')
            
        Pmatrix = Pm[matrix1]
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
        
        #AvgBessel = np.array([ avgBessel(l, k ,rmin, rmax) for k in kbin ])#/Vir
        AvgBessel = avgBessel(l, kmatrix, rminmatrix, rmaxmatrix )
        #AvgBessel = sbess(l, kmatrix * rmatrix)
        multipole_xi = np.real(I**l) * simpson(kmatrix**2 * Pmatrix * AvgBessel/(2*np.pi**2), kbin, axis=0)#/Vir

        return multipole_xi
    
    
    def fourier_transform_kr(self, l, kbin, p):
        """
        Calculate Fourier transform k -> r for a given l value
        kbin and p should be very fine in log scale 
        otherwise the result would be quite noisy.

        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """

	
        import cmath
        I = cmath.sqrt(-1)
        if self.nn == 0 : overnn = 0
        else : overnn = 1./self.nn
            
        #kbin = self.kbin
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
	
	#Pinterp = interp1d(kbin, p)
	#Pinterp = Pinterp(k_samp)

        matrix1,matrix2 = np.mgrid[0:len(kbin),0:len(rcenter)]
        kmatrix = kbin[matrix1]
        
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        rmatrix = rcenter[matrix2]
        Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)
        
        #try : self.Pmultipole0_interp
        #except : self.multipole_P_band_all()
            
        if l == 0 : 
            try : Pm = p - overnn
            except (ZeroDivisionError): Pm = p
        elif l == 2 : Pm = p
        elif l == 4 : Pm = p
        else : raise ValueError('l should be 0, 2, 4')
            
        Pmatrix = Pm[matrix1]
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
        
        #AvgBessel = np.array([ avgBessel(l, k ,rmin, rmax) for k in kbin ])#/Vir
        AvgBessel = avgBessel(l, kmatrix, rminmatrix, rmaxmatrix )
        #AvgBessel = sbess(l, kmatrix * rmatrix)
        multipole_xi = np.real(I**l) * simpson(kmatrix**2 * Pmatrix * AvgBessel/(2*np.pi**2), kbin, axis=0)#/Vir

        return multipole_xi
    

    def fourier_transform_kr1r2(self, l1, l2, kbin, p):

        """
        Calculate Double Bessel Fourier transform k -> r1, r2 for a given l value
        \int k^2/2pi^2 j0(kr1) j0(kr2) p(k)

        kbin and p should be very fine in log scale 
        otherwise the result will be quite noisy.

        Parameters
        ----------
        l : mode (0, 2, 4)
        
        """        
        from fortranfunction import sbess
        sbess = np.vectorize(sbess)
                
        import cmath
        I = cmath.sqrt(-1)
        
        if self.nn == 0: overnn = 0.0
        else : overnn = 1./self.nn
        #shotnoise_term = (2*l1 + 1.) * 2. * (2 * pi)**3/self.Vs*overnn**2 /(4*np.pi*kbin**2)

        #klist = self.kbin_y
        #kbin = self.kbin

        rcenter = self.rcenter
        #dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)
        #matrix1, matrix2, matrix3 = np.mgrid[ 0:kbin.size, 0: rcenter.size, 0:rcenter.size]
        matrix1, matrix2 = np.mgrid[ 0:kbin.size, 0: rcenter.size]
        kmatrix = kbin[matrix1]
        rmatrix = rcenter[matrix2]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        #Besselmatrix = sbess(l1, kmatrix * rmatrix)
        #Besselmatrix = np.array([ avgBessel(l1, k ,rmin, rmax) for k in kbin ]) #/Vir
        Besselmatrix1 = avgBessel(l1, kmatrix, rminmatrix, rmaxmatrix )
        Besselmatrix2 = avgBessel(l2, kmatrix, rminmatrix, rmaxmatrix )


        Cll = np.real(I**(l1+l2)) /(2*np.pi)**3 * p
        #Cxillmatrix = Cll[matrix1] * Besselmatrix1 *  Besselmatrix2
        
        
        i=0
        Cxill_matrix = np.zeros((rcenter.size, rcenter.size))
        for ri in range(rcenter.size):
            for rj in range(rcenter.size):
                cxill = Cll * Besselmatrix1[:,ri] * Besselmatrix2[:,rj] * kbin**2/(2*np.pi**2)
                Cxill_matrix[ri,rj] = simpson( cxill, kbin )
                print 'cov xi {}/{} \r'.format(i, rcenter.size**2),
                i+=1
        """
        i = 0
        m1, m2 = np.mgrid[0:kbin.size, 0:rcenter.size]
        Cxill_matrix = np.zeros((rcenter.size, rcenter.size))
        for ri in range(rcenter.size):
            #for rj in range(rcenter.size):
            cxill = Cll * Besselmatrix[:,ri][m1] * Besselmatrix * kbin[m1]**2/(2*np.pi**2)
            Cxill_matrix[:,i] = simpson( cxill, kbin, axis = 0 )
            print '{}/{} \r'.format(i, rcenter.size),
            i+=1
        """
        return Cxill_matrix         
            
    def covariance_PP(self, l1, l2):

        #from scipy.integrate import simps, romb
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        kbin = self.kbin
        #kcenter = self.kcenter_y
        #skbin = self.skbin
        mulist = self.mulist
        #dk = self.kmax_y - self.kmin_y
        #dlnk = self.dlnk
        #sdlnk = self.sdlnk
        #sdk = self.skmax - self.skmin
        PS = self.Pm_interp(kbin)
        
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        dmu = self.dmu
        
        if self.nn == 0: overnn = 0.0
        else : overnn = 1./self.nn
            
        # FirstTerm + SecondTerm
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kbin.size]
        mumatrix = self.mulist[matrix1]
        
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        
        
        Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) * (2*pi)**3 /Vs
        
        kmatrix = kbin[matrix2]
        Pmmatrix = PS[matrix2]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0 : Dmatrix = 1.
        R = (b + f * mumatrix**2)**2 * Dmatrix
        
        Rintegral3 = Const_alpha * PS**2 * romberg( R**2 * Le_matrix1 * Le_matrix2, dx=dmu, axis=0 )/(4*np.pi*kbin**2)
        Rintegral2 = Const_alpha * 2.*overnn * PS * romberg( R * Le_matrix1 * Le_matrix2, dx=dmu, axis=0 )/(4*np.pi*kbin**2)
     

        FirstSecond = Rintegral3 + Rintegral2
        if self.nn == 0 : FirstSecond = Rintegral3
        
        # LastTerm
        if l1 == l2:
            LastTerm = (2*l1 + 1.) * 2. * (2 * pi)**3/Vs*overnn**2 /(4*np.pi*kbin**2)
        else:
            LastTerm = np.zeros(FirstSecond.shape)
        
        if self.nn == 0 : LastTerm = np.zeros(FirstSecond.shape)
        
        Total = FirstSecond + LastTerm
        #Total = LastTerm
        
        if l1 == 0 : 
            if l2 == 0 : 
                self.covP00_interp = log_interp(kbin, Total)
                covP_interp = self.covP00_interp
                 
            elif l2 == 2 :
                self.covP02_interp = log_interp(kbin, Total)
                covP_interp = self.covP02_interp
            else : 
                self.covP04_interp = log_interp(kbin, Total)
                covP_interp = self.covP04_interp
                
        elif l1 == 2 : 
            if l2 == 2 :
                self.covP22_interp = log_interp(kbin, Total)
                covP_interp = self.covP22_interp
            elif l2 == 4 :
                self.covP24_interp = log_interp(kbin, Total)  
                covP_interp = self.covP24_interp
            else : raise ValueError
                
        elif l1 == 4 : 
            if l2 == 4 :
                self.covP44_interp = log_interp(kbin, Total)   
                covP_interp = self.covP44_interp
            else : raise ValueError
        
        """
        Vi = 4./3 * pi * ( self.kmax_y**3 - self.kmin_y**3 ) # ~ 4*np.pi*kcenter**2*dk
        volume_fac = (4*np.pi*kcenter**2)/Vi
        covP_diag = covP_interp(kcenter) * volume_fac  
  
        covariance_mutipole_PP = np.zeros((kcenter.size,kcenter.size))
        np.fill_diagonal(covariance_mutipole_PP,covP_diag)
        """      
        covP_diag = covP_interp(self.kbin)/(self.kbin * self.dlnk)
        covariance_mutipole_PP = np.zeros((self.kbin.size,self.kbin.size))
        np.fill_diagonal(covariance_mutipole_PP,covP_diag)
        
        return covariance_mutipole_PP




    def __covariance_PP(self, l1, l2):

        if l1 == 0 : 
            if l2 == 0 : 
                self.covP00_interp = log_interp(kbin, Total)
                covP_interp = self.covP00_interp
                 
            elif l2 == 2 :
                self.covP02_interp = log_interp(kbin, Total)
                covP_interp = self.covP02_interp
            else : 
                self.covP04_interp = log_interp(kbin, Total)
                covP_interp = self.covP04_interp
                
        elif l1 == 2 : 
            if l2 == 2 :
                self.covP22_interp = log_interp(kbin, Total)
                covP_interp = self.covP22_interp
            elif l2 == 4 :
                self.covP24_interp = log_interp(kbin, Total)  
                covP_interp = self.covP24_interp
            else : raise ValueError
                
        elif l1 == 4 : 
            if l2 == 4 :
                self.covP44_interp = log_interp(kbin, Total)   
                covP_interp = self.covP44_interp
            else : raise ValueError

        covP_diag = covP_interp(self.kbin)/(self.kbin * self.dlnk)
        covariance_mutipole_PP = np.zeros((self.kbin.size,self.kbin.size))
        np.fill_diagonal(covariance_mutipole_PP,covP_diag)
        return covariance_mutipole_PP

    
    
    def covariance_Xi(self, l1, l2):

        """
        Covariance Xi(r1, r2)

        Calculate cov_xi matrix from cov_p by double bessel fourier transform. 
        self.covariance_PP(l1, l2) should be called first. 

        """


        kbin = self.kbin

        try : self.covP00_interp(1.0)
        except : self.covariance_PP(l1, l2)
            
        if l1 == 0 :
            if l2 == 0 : 
                Cll = self.covP00_interp(kbin)# -covPP_LastTerm
            elif l2 == 2 : 
                Cll = self.covP02_interp(kbin)
            else : 
                Cll = self.covP04_interp(kbin)
        elif l1 == 2 :
            if l2 == 0 : 
                Cll = self.covP02_interp(kbin)
            elif l2 == 2 : 
                Cll = self.covP22_interp(kbin)# -covPP_LastTerm
            elif l2 == 4 : 
                Cll = self.covP24_interp(kbin)
            else : raise ValueError
        elif l1 == 4 :
            if l2 == 0 : 
                Cll = self.covP04_interp(kbin)
            elif l2 == 2 : 
                Cll = self.covP24_interp(kbin)
            elif l2 == 4 : 
                Cll = self.covP44_interp(kbin)# -covPP_LastTerm
            else : raise ValueError
        else : raise ValueError('l should be 0,2,4')


        covp_diag = Cll *4*np.pi*kbin**2
        covxi = self.fourier_transform_kr1r2(l1, l2, kbin, covp_diag)

        return covxi


    def covariance_PXi(self, l1, l2):

        """
        This function calculates covariance PXi(l1, l2).
        self.covariance_PP(l1,l2) should be called first.
        l1 : P
        l2 : Xi

        """

        kbin = self.kbin
        #kbin = self.kcenter_y
        rbin = self.rcenter
        rmin = self.rmin
        rmax = self.rmax
        #Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)

        try : self.covP00_interp(1.0)
        except : self.covariance_PP(l1, l2)
            
        if l1 == 0 :
            if l2 == 0 : 
                Cll = self.covP00_interp(kbin)# -covPP_LastTerm
            elif l2 == 2 : 
                Cll = self.covP02_interp(kbin)
            else : 
                Cll = self.covP04_interp(kbin)
        elif l1 == 2 :
            if l2 == 0 : 
                Cll = self.covP02_interp(kbin)
            elif l2 == 2 : 
                Cll = self.covP22_interp(kbin)# -covPP_LastTerm
            elif l2 == 4 : 
                Cll = self.covP24_interp(kbin)
            else : raise ValueError
        elif l1 == 4 :
            if l2 == 0 : 
                Cll = self.covP04_interp(kbin)
            elif l2 == 2 : 
                Cll = self.covP24_interp(kbin)
            elif l2 == 4 : 
                Cll = self.covP44_interp(kbin)# -covPP_LastTerm
            else : raise ValueError
        else : raise ValueError('l should be 0,2,4')


        #from fortranfunction import sbess
        #sbess = np.vectorize(sbess)
        matrix1, matrix2 = np.mgrid[0:kbin.size, 0:rbin.size]
        kmatrix = kbin[matrix1]
        rmatrix = rbin[matrix2]
        rminmatrix = rmin[matrix2]
        rmaxmatrix = rmax[matrix2]
        #Besselmatrix = np.array([ avgBessel(l2, k ,rmin, rmax) for k in kbin ])#/Vir
        Besselmatrix = avgBessel(l2, kmatrix, rminmatrix, rmaxmatrix )
        #Besselmatrix = sbess(l2, kmatrix * rmatrix)

        covp_diag = Cll * (4*np.pi*kbin**2)
        Cllmatrix = covp_diag[matrix1] * Besselmatrix /(2*np.pi)**3

        return Cllmatrix




    def derivative_bfs(self,l):
        """
        
        Calculate dXi/d(params) vector.
        self.deriative_P_bfs(l) should be called first.        

        Parameters
        ----------
        l : mode (0, 2, 4)
        
        output
        -------
        dxi/db, dxi/df, dxi/ds for a given l value.


        """
        import cmath
        I = cmath.sqrt(-1)
        if self.nn == 0 : overnn = 0
        else : overnn = 1./self.nn

        kbin = self.kbin
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
        #Vir = 4./3 * np.pi * np.fabs(rmax**3 - rmin**3)
        
        try : self.dPdb0_interp
        except (AttributeError) : self.derivative_P_bfs_all()
        if l == 0 : 
            dpdb, dpdf, dpds = self.dPdb0_interp(kbin), self.dPdf0_interp(kbin), self.dPds0_interp(kbin)
        elif l == 2 : 
            dpdb, dpdf, dpds = self.dPdb2_interp(kbin), self.dPdf2_interp(kbin), self.dPds2_interp(kbin)
        elif l == 4 : 
            dpdb, dpdf, dpds = self.dPdb4_interp(kbin), self.dPdf4_interp(kbin), self.dPds4_interp(kbin)
        else : raise ValueError('l should be 0, 2, 4')
            
        dpdb = dpdb[matrix1]
        dpdf = dpdf[matrix1]
        dpds = dpds[matrix1]
        
        from fortranfunction import sbess
        #sbess = np.vectorize(sbess)
        
        #AvgBessel = np.array([ avgBessel(l, k ,rmin, rmax) for k in kbin ])#/Vir
        AvgBessel = avgBessel(l, kmatrix, rminmatrix, rmaxmatrix )#/Vir
        #AvgBessel = sbess(l, kmatrix * rmatrix)
        self.dxidb = np.real(I**l) * simpson(kmatrix**2 * dpdb * AvgBessel/(2*np.pi**2), kbin, axis=0)#/Vir
        self.dxidf = np.real(I**l) * simpson(kmatrix**2 * dpdf * AvgBessel/(2*np.pi**2), kbin, axis=0)#/Vir
        self.dxids = np.real(I**l) * simpson(kmatrix**2 * dpds * AvgBessel/(2*np.pi**2), kbin, axis=0)#/Vir


        return self.dxidb, self.dxidf, self.dxids
 


  

    def derivative_P_bfs(self,l):
        
        """
        
        Calculate dP/d(params) vector.      

        Parameters
        ----------
        l : mode (0, 2, 4)
        
        output
        -------
        dp/db, dp/df, dp/ds for a given l value.


        """
  
        b = self.b
        f = self.f
        s = self.s
        
        kbin = self.kbin
        #kcenter = self.kcenter_y
        #skbin = self.skbin
        #skcenter = self.skcenter
        #dk = self.dk
        dmu = self.dmu
        mulist = self.mulist
        #dlnk = self.dlnk
        #Pmlist = self.Pmlist
        Pm = self.Pm_interp(kbin)
        
        matrix1, matrix2 = np.mgrid[0:mulist.size,0:kbin.size]
        
        kmatrix = kbin[matrix2]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mulist)[matrix1]
        
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        #Vi = 4./3 * pi * (self.kmax_y**3 - self.kmin_y**3)
        Pmmatrix = Pm[matrix2]
        Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
        if self.s == 0:Dmatrix = 1.
            
        Rb = 2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
        Rf = 2 * mumatrix**2 *(self.b + self.f * mumatrix**2) * Dmatrix #* Le_matrix
        Rs = 2 *self.s*(- kmatrix**2 * mumatrix**2)*(self.b + self.f * mumatrix**2)**2 * Dmatrix# * Le_matrix
        Rintb = (2 * l + 1.)/2 * romberg( Pmmatrix * Rb * Le_matrix, dx=dmu, axis=0 )
        Rintf = (2 * l + 1.)/2 * romberg( Pmmatrix * Rf * Le_matrix, dx=dmu, axis=0 )
        Rints = (2 * l + 1.)/2 * romberg( Pmmatrix * Rs * Le_matrix, dx=dmu, axis=0 )

        dPdb_interp = interp1d(kbin, Rintb)
        dPdf_interp = interp1d(kbin, Rintf)
        dPds_interp = interp1d(kbin, Rints)
        
        if l == 0 : 
            self.dPdb0_interp, self.dPdf0_interp, self.dPds0_interp = dPdb_interp, dPdf_interp, dPds_interp
        elif l == 2 : 
            self.dPdb2_interp, self.dPdf2_interp, self.dPds2_interp = dPdb_interp, dPdf_interp, dPds_interp
        elif l == 4 : 
            self.dPdb4_interp, self.dPdf4_interp, self.dPds4_interp = dPdb_interp, dPdf_interp, dPds_interp
        else : raise ValueError('l should be 0, 2, 4')
            
        #return dPdb_interp(kcenter), dPdf_interp(kcenter), dPds_interp(kcenter)
        return dPdb_interp(kbin), dPdf_interp(kbin), dPds_interp(kbin)
    

        

        
