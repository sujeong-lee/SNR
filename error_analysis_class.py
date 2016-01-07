import numpy as np
from numpy import zeros, sqrt, pi, sin, cos, exp
from numpy.linalg import pinv, inv
from numpy import vectorize
from scipy.interpolate import interp1d
from scipy.integrate import simps, romb

class Linear_covariance():
    
    """ No redshift"""

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN, N_x):
        
        #parameter
        self.h= 1.0
        self.Vs= 5.0*10**9 # survey volume
        self.nn= 3.0 * 10**(-4) # shot noise : \bar{n}
        
        # k scale range
        self.KMIN = KMIN
        self.KMAX = KMAX
        
        # r scale
        self.RMIN = RMIN #30.#0.1 #0.1 for Reid   #1.15 * np.pi / self.kmax
        self.RMAX = RMAX #180. #180. for Reid  #1.15 * np.pi / self.kmin
        
        self.n = n # converge # 201 for Reid # 201 # the number of k bins. the sample should be odd
        self.n2 = n2 #101 for Reid # number of r bins
        self.subN = subN
        #self.klist = np.logspace(np.log(self.KMIN),np.log(self.KMAX),self.n, base = np.e)
        
        
        self.rlist = np.logspace(np.log(self.RMIN),np.log(self.RMAX),self.n2, base = np.e)
        rlist = self.rlist
        self.rmin = np.delete(rlist,-1)
        self.rmax = np.delete(rlist,0)
        
        #self.rcenter = np.array([ (rlist[i] + rlist[i+1])/2. for i in range(len(rlist)-1) ])
        self.rcenter = np.array([ np.sqrt(rlist[i] * rlist[i+1]) for i in range(len(rlist)-1) ])
        self.dr = np.fabs(self.rmax - self.rmin)
        self.dlnr = np.fabs(np.log(self.rcenter[2]/self.rcenter[3]))# np.log(self.rmax/self.rmin)[3]
        
   
   
        self.skbin = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), subN * self.n + 1, base=10)
        self.skmin = np.delete(self.skbin,-1)
        self.skmax = np.delete(self.skbin,0)
        #self.skcenter = np.array([(self.skbin[i] + self.skbin[i+1])/2. for i in range(len(self.skbin)-1)])
        self.skcenter = np.array([(np.sqrt(self.skbin[i] * self.skbin[i+1])) for i in range(len(self.skbin)-1)])
        self.sdlnk = np.log(self.skcenter[3]/self.skcenter[2]) #np.log(self.skmax/self.skmin)
        
        self.sdk = self.skmax - self.skmin
        self.klist = np.array([self.skbin[i*subN] for i in range(len(self.skbin)/subN + 1)]) #For Reid, delete '+1', not relevant anymore
        
        #self.kcenter = np.array([(self.klist[i] + self.klist[i+1])/2. for i in range(len(self.klist)-1)])
        self.kcenter = np.array([self.skcenter[i*subN + subN/2] for i in range(len(self.skcenter)/subN)])
        self.kmin = np.delete(self.klist,-1)
        self.kmax = np.delete(self.klist,0)
        self.dk = self.kmax - self.kmin
        self.dlnk = np.log(self.kcenter[3]/self.kcenter[2]) #np.log(self.kmax/self.kmin)[3]


        # k bin for xi integral ----------
        self.N_x = N_x #2**18 + 1
        self.kbin_x = np.logspace(np.log10(self.KMIN),np.log10(self.KMAX), self.N_x + 1, base=10) # for romb int, +2
        self.kcenter_x = np.array([(np.sqrt(self.kbin_x[i] * self.kbin_x[i+1])) for i in range(len(self.kbin_x)-1)])
        self.kmin_x = np.delete(self.kbin_x,-1)
        self.kmax_x = np.delete(self.kbin_x,0)
        self.dk_x = self.kmax_x - self.kmin_x
        self.dlnk_x = np.log(self.kcenter_x[3]/self.kcenter_x[2]) #np.log(self.kmax_x/self.kmin_x)
    

        InitiateTitle = '-------------------------------------------------------------------\
        \nclass error_analysis, no RSD \
        \nz = 0.0, kN ={}, subN = {}, rN = {}, N_x = {} \
        \ndlnr = {}, dlnk={}, sdlnk={}'.format(self.n, self.subN, self.n2, self.N_x, self.dlnr, self.dlnk, self.sdlnk )
    
        #print InitiateTitle


    def compile_fortran_modules(self):
        
        import numpy.f2py.f2py2e as f2py2e
        import sys
        
        sys.argv +=  "-c -m fortranfunction fortranfunction.f90".split()
        f2py2e.main()
        sys.argv = [sys.argv[0]]
        
        
        #sys.argv +=  "-c -m sici sici.f90".split()
        #f2py2e.main()
        #sys.argv = [sys.argv[0]]
        #sys.argv +=  "-c -m legen legen.f90".split()
        #f2py2e.main()
        #sys.argv = [sys.argv[0]]
        

    def MatterPower(self, file):
        
        #Pkl=np.array(np.loadtxt('matterpower_z_0.55.dat')) # z=0.55
        Pkl=np.array(np.loadtxt(file))
        k=np.array(Pkl[:,0])
        P=np.array(Pkl[:,1])
        
        #power spectrum interpolation
        Pm = interp1d(k, P ,kind= "cubic") #matterpower
        #self.Pmlist = Pm(self.kcenter)
        
        #REAL POWERSPECTRUM DATA
        self.RealPowerBand = np.array([Pm(self.skcenter[i]) for i in range(len(self.skcenter))])
        self.RealPowerBand_x = np.array([Pm(self.kcenter_x[i]) for i in range(len(self.kcenter_x))])


    def Shell_avg_band( self ):
        #
        #   Shell_avg_band
        #
        from scipy.integrate import simps
        powerspectrum = self.RealPowerBand
        skbin = self.skbin
        skcenter = self.skcenter
        kcenter = self.kcenter
        dk = self.dk
        kmax = self.kmax
        kmin = self.kmin
        
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        Vi = 4./3 * pi * (kmax**3 - kmin**3)
        
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN:i*self.subN+self.subN]
            data = powerspectrum[i*self.subN:i*self.subN+self.subN]
            
            result = simps(4* np.pi * k**2 * data, k )
            resultlist.append(result)
        self.Pmlist = resultlist/Vi
        return self.Pmlist





class RSD_covariance(Linear_covariance):

    def __init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN, N_x):
        Linear_covariance.__init__(self, KMIN, KMAX, RMIN, RMAX, n, n2, subN, N_x)
        
        # RSD parameter

        self.b=2.0
        self.f=0.74
        self.s= 3.5  # sigma in Power spectrum

        self.n3 = 2**6 + 1 # 101 for Reid number of mu bins
        self.mulist = np.linspace(-1.,1.,self.n3)
        self.dmu = self.mulist[3]-self.mulist[2]
        
        InitiateTitle = '\nclass RSD_covariance \
        \nz = 0.55, kN ={}, subN = {}, rN = {}, N_x = {} \
        \ndlnr = {}, dlnk={}, sdlnk={} \ndr = ({},{}), \
        dk = ({},{})'.format(self.n, self.subN, self.n2, self.N_x, self.dlnr ,self.dlnk, self.sdlnk, self.dr[0], self.dr[-1], self.dk[0], self.dk[-1] )
        print InitiateTitle


    def multipole_P(self,l):
    
        b = self.b
        f = self.f
        s = self.s

        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        mulist = self.mulist
        dlnk = self.dlnk
        Pmlist = 1. #self.Shell_avg_band()
        matterpower = self.RealPowerBand
        
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        Le_matrix = Ll(l,mumatrix)

        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        Vi = 4./3 * pi * (self.kmax**3 - self.kmin**3)
                
        resultlist=[]
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Pmmatrix = Pm[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (b + f * mumatrix**2)**2 * Dmatrix * Le_matrix
            muint = (2 * l + 1.)/2 * simps( 4 * pi * kmatrix**2 * Pmmatrix * R, mumatrix, axis=0 )
            result = simps( muint, k )
            resultlist.append(result)
        return resultlist/Vi
    
    
    def multipole_P_band_all(self):
    
        self.multipole_bandpower0 = self.multipole_P(0.0)
        self.multipole_bandpower2 = self.multipole_P(2.0)
        self.multipole_bandpower4 = self.multipole_P(4.0)
        self.multipole_bandpower = np.concatenate((self.multipole_bandpower0, self.multipole_bandpower2,self.multipole_bandpower4), axis=0)
        

    def derivative_Xi_band(self, l):
        #
        #   Shell averaged
        #   dxi_l / dp_li = i^l int(k^2 ShellavgBessel(kr) / 2pi^2, kmin, kmax)
        #
        import numpy as np
        from numpy import pi
        from scipy.integrate import simps, romb
        import cmath
        I = cmath.sqrt(-1)

        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        skbin = self.skbin
        dk = self.dk
        sdlnk = self.sdlnk
        mulist = self.mulist
        dlnk = self.dlnk
        rcenter = self.rcenter
        rmin = self.rmin
        rmax = self.rmax
        dr = self.dr
        
        matrix1, matrix2 = np.mgrid[ 0:len(kcenter), 0: len(rcenter) ]
        matrix3, matrix4 = np.mgrid[0: self.subN, 0:len(rcenter)]
        
        rminmatrix = rmin[matrix4]
        rmaxmatrix = rmax[matrix4]
        rmatrix = rcenter[matrix4]

        #Vir = 4 * pi * rcenter**2 * dr + 1./3 * pi * dr**3
        Vir = 4./3 * pi * np.fabs(rmax**3 - rmin**3)
        
        resultlist=[]
        for i in range(len(kcenter)):
            k = self.skcenter[i*self.subN : i*self.subN + self.subN ]
            kmatrix2 = k[matrix3]
            avgBesselmatrix = avgBessel(l, kmatrix2 ,rminmatrix,rmaxmatrix)
            result = np.real(I**l) * romb( kmatrix2**3/(2*pi**2) * avgBesselmatrix  , dx = sdlnk , axis = 0 )/Vir
            resultlist.append(result)
        
        derivative_Xi_bandpower = np.array(resultlist)
        return derivative_Xi_bandpower
            
            
    def derivative_Xi_band_all(self):
    
        import pp, sys, time
        
        ppservers = ()
        
        if len(sys.argv) > 1:
            ncpus = int(sys.argv[1])
            # Creates jobserver with ncpus workers
            job_server = pp.Server(ncpus, ppservers=ppservers)
        else:
            # Creates jobserver with automatically detected number of workers
            job_server = pp.Server(ppservers=ppservers)
        #print "Starting pp with", job_server.get_ncpus(), "workers"
        
        inputs1 = ((0.0,),(2.0,),(4.0,))
        jobs1 = [ job_server.submit(self.derivative_Xi_band, input, (avgBessel,)) for input in inputs1]
        result1=[]
        for job in jobs1:
            re = job()
            result1.append(re)

        self.dxip0 = result1[0]
        self.dxip2 = result1[1]
        self.dxip4 = result1[2]
            
        print "derivative_Xi_band_all (dxi/dp)"
    

    def RSDband_covariance_PP(self, l1, l2):
        
        from scipy.integrate import quad,simps
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter
        skcenter = self.skcenter
        mulist = self.mulist
        dk = self.dk
        dlnk = self.dlnk
        sdlnk = self.sdlnk
        matterpower = self.RealPowerBand
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        dmu = self.dmu
        
        # FirstTerm + SecondTerm
        matrix1, matrix2 = np.mgrid[0:len(mulist),0:self.subN]
        mumatrix = self.mulist[matrix1]
        
        Le_matrix1 = Ll(l1,mumatrix)
        Le_matrix2 = Ll(l2,mumatrix)
        #Vi = 4 * pi * kcenter**2 * dk + 1./3 * pi * (dk)**3
        Vi = 4./3 * pi * ( self.kmax**3 - self.kmin**3 )
        
        Const_alpha = (2*l1 + 1.) * (2*l2 + 1.) * (2*pi)**3 /Vs
       
        resultlist1 = []
        resultlist2 = []
        for i in range(len(kcenter)):
            k = skcenter[i*self.subN : i*self.subN+self.subN]
            Pm = matterpower[i*self.subN : i*self.subN+self.subN]
            kmatrix = k[matrix2]
            Dmatrix = np.exp(- kmatrix**2 * mumatrix**2 * self.s**2) #FOG matrix
            R = (self.b + self.f * mumatrix**2)**2 * Dmatrix
            Rintegral3 =  simps(  R**2 * Le_matrix1 * Le_matrix2, dx = dmu, axis=0 )
            Rintegral2 =  simps(  R * Le_matrix1 * Le_matrix2, dx = dmu, axis=0 )
            result1 = simps( 4 * pi * k**3 * Pm**2 * Rintegral3, dx = sdlnk )
            result2 = simps( 4 * pi * k**3 * Pm * Rintegral2, dx = sdlnk )
            resultlist1.append(result1)
            resultlist2.append(result2)
        FirstTerm = Const_alpha * np.array(resultlist1)/Vi**2
        SecondTerm = Const_alpha * 2./nn * np.array(resultlist2)/Vi**2
        
        # LastTerm
        
        if l1 == l2:
            LastTerm = (2*l1 + 1.) * 2. * (2 * pi)**3/Vs/nn**2 /Vi
        else:
            LastTerm = 0.
        
        Total = FirstTerm + SecondTerm + LastTerm
        covariance_mutipole_PP = np.zeros((len(kcenter),len(kcenter)))
        np.fill_diagonal(covariance_mutipole_PP,Total)

        #print 'covariance_PP {:>1.0f}{:>1.0f} is finished'.format(l1,l2)
        return covariance_mutipole_PP
  
  
    def RSDband_covariance_PP_all(self):
        
        self.covariance_PP00 = np.array(self.RSDband_covariance_PP(0.0,0.0))
        self.covariance_PP02 = np.array(self.RSDband_covariance_PP(0.0,2.0))
        self.covariance_PP04 = np.array(self.RSDband_covariance_PP(0.0,4.0))
        self.covariance_PP22 = np.array(self.RSDband_covariance_PP(2.0,2.0))
        self.covariance_PP24 = np.array(self.RSDband_covariance_PP(2.0,4.0))
        self.covariance_PP44 = np.array(self.RSDband_covariance_PP(4.0,4.0))
  
        print 'covariance_PP_all is done'
    
  


    def RSDband_covariance_Xi_all(self):
    
        """ Output : submatrices C_ll' for each modes (l,l' = 0,2,4)
                size of each matrix is (# of r bins) x (# of r bins)
    
           C_ll' = <X_l(ri)X_l'(rj)> """

        from scipy.integrate import simps, romb
        from numpy import zeros, sqrt, pi, exp
        import cmath
        I = cmath.sqrt(-1)
    
        klist = self.klist
        kcenter = self.kcenter_x
        skcenter = self.kcenter_x
        rlist = self.rlist
        rcenter = self.rcenter
        dr = self.dr
        rmin = self.rmin
        rmax = self.rmax
        mulist = self.mulist
        dk = self.dk_x
        dlnk = self.dlnk_x
        sdlnk = self.dlnk_x
        dr = self.dr
        Pmlist = self.Pmlist
        Pm = self.RealPowerBand_x
        s = self.s
        b = self.b
        f = self.f
        Vs = self.Vs
        nn = self.nn
        
    
        # generating 2-dim matrix for k and mu, matterpower spectrum, FoG term
        matrix1,matrix2 = np.mgrid[0:len(mulist),0:len(skcenter)]
        mulistmatrix = mulist[matrix1] # mu matrix (axis 0)
        klistmatrix = skcenter[matrix2] # k matrix (axis 1)
        Le_matrix0 = Ll(0.0,mulistmatrix)
        Le_matrix2 = Ll(2.0,mulistmatrix)
        Le_matrix4 = Ll(4.0,mulistmatrix)
    
        Dmatrix = np.exp(-klistmatrix**2 * mulistmatrix**2 * self.s**2)
        R = (b + f * mulistmatrix**2)**2 * Dmatrix

        from multiprocessing import Process, Queue
        
        """print 'Rintegral' """
        def Rintegral(q, order, (l1, l2, Le1, Le2)):
            
            #import covariance_class2
            from numpy import pi, real
            from scipy.integrate import simps
            import cmath
            
            I = cmath.sqrt(-1)
            const_gamma = real(I**(l1+l2)) * 2.* (2*l1+1)*(2*l2+1) /(2*pi)**2 /self.Vs
            Rintegral3 = romb(R**2 * Le1 * Le2, dx = self.dmu, axis=0 )
            Rintegral2 = romb(R * Le1 * Le2, dx = self.dmu, axis=0 )
            result = const_gamma * skcenter**3 * (Rintegral3 * Pm**2 + Rintegral2 * Pm * 2./self.nn)
            
            q.put((order,result))
        
        inputs = (( 0.0, 0.0, Le_matrix0, Le_matrix0),( 0.0, 2.0, Le_matrix0, Le_matrix2),(0.0, 4.0,Le_matrix0, Le_matrix4),(2.0, 2.0, Le_matrix2, Le_matrix2),(2.0, 4.0, Le_matrix2, Le_matrix4),(4.0, 4.0, Le_matrix4, Le_matrix4))
        
        R_queue = Queue()
        R_processes = [Process(target=Rintegral, args=(R_queue, z[0], z[1])) for z in zip(range(6), inputs)]
        for p in R_processes:
            p.start()
        #for p in R_processes:
            #p.join()
        Rintegrals = [R_queue.get() for p in R_processes]
            #for p in R_processes:
            #p.terminate()

        Rintegrals.sort()
        Rintegrallist = [R[1] for R in Rintegrals]
        
        Rintegral00 = Rintegrallist[0] # 1D
        Rintegral02 = Rintegrallist[1]
        Rintegral04 = Rintegrallist[2]
        Rintegral22 = Rintegrallist[3]
        Rintegral24 = Rintegrallist[4]
        Rintegral44 = Rintegrallist[5]
    
        matrix4,matrix5 = np.mgrid[0:len(rcenter),0:len(rcenter)]
        rlistmatrix1 = rcenter[matrix4] # vertical
        rlistmatrix2 = rcenter[matrix5] # horizontal
        dr1 = dr[matrix4] # vertical
        dr2 = dr[matrix5] # horizontal
        rminmatrix = rmin[matrix4] # vertical
        rminmatrix2 = rmin[matrix5] # horizontal
        rmaxmatrix = rmax[matrix4] # vertical
        rmaxmatrix2 = rmax[matrix5] # horizontal

        Vir1 = 4./3 * pi * np.fabs(rminmatrix**3 - rmaxmatrix**3)
        Vir2 = 4./3 * pi * np.fabs(rminmatrix2**3 - rmaxmatrix2**3)
        Vi = 4./3 * pi * np.fabs(rmin**3 - rmax**3)
        
        def AvgBessel_q(q, order, (l, skcenter, rmin, rmax)):
            Avg = [avgBessel(l,k,rmin,rmax) for k in skcenter] #2D (kxr)
            q.put((order,Avg))
    
        
        inputs_bessel = [(0.0, skcenter,rmin,rmax),(2.0, skcenter,rmin,rmax), (4.0, skcenter,rmin,rmax) ]

        B_queue = Queue()
        B_processes = [Process(target=AvgBessel_q, args=(B_queue,z[0], z[1])) for z in zip(range(3), inputs_bessel)]
        for pB in B_processes:
            pB.start()
        Bessels = [B_queue.get() for pB in B_processes]
        Bessels.sort()
        Bessel_list = [ B[1] for B in Bessels] #2D bessel, (kxr)


        avgBesselmatrix0 = np.array(Bessel_list[0]) #2D, (kxr)
        avgBesselmatrix2 = np.array(Bessel_list[1])
        avgBesselmatrix4 = np.array(Bessel_list[2])

        matrix1, matrix2 = np.mgrid[0:len(skcenter), 0:len(rcenter)]
        Volume_double = Vir1 * Vir2


        def FirstSecond(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
        
            Rint_result = result[matrix1] # 2D
            
            relist = []
            for i in range(len(rcenter)/2):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                re = romb(Rint_result * avgBmatrix * avgBessel2, dx = sdlnk, axis=0)
                relist.append(re)
            FirstTerm = relist/ Volume_double[0:len(rcenter)/2,:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[0:len(rcenter)/2,:]
            else : LastTerm = 0.
            
        
            re = FirstTerm+LastTerm
            queue.put((order,re))


        def FirstSecond2(queue, order, (l1, l2, result, avgBessel1, avgBessel2)):
    
            Rint_result = result[matrix1] # 2D
    
            relist = []
            for i in range(len(rcenter)/2, len(rcenter)):
                avgBmatrix = np.array(avgBessel1[:, i])[matrix1]
                re = romb(Rint_result * avgBmatrix * avgBessel2, dx = sdlnk, axis=0)
                relist.append(re)
            FirstTerm = relist/ Volume_double[len(rcenter)/2:len(rcenter),:] #2D
            if l1 == l2:
                Last = (2./Vs) * (2*l1+1)/nn**2 / Vi #1d array
                LastTermmatrix = np.zeros((len(rcenter),len(rcenter)))
                np.fill_diagonal(LastTermmatrix,Last)
                LastTerm = LastTermmatrix[len(rcenter)/2:len(rcenter),:]
                
            else : LastTerm = 0.
            
            re = FirstTerm+LastTerm
            queue.put((order,re))

        
        F_inputs = (( 0.0, 0.0, Rintegral00, avgBesselmatrix0, avgBesselmatrix0),( 0.0, 2.0, Rintegral02,  avgBesselmatrix0, avgBesselmatrix2),(0.0, 4.0, Rintegral04, avgBesselmatrix0, avgBesselmatrix4 ),(2.0, 2.0, Rintegral22, avgBesselmatrix2, avgBesselmatrix2 ),(2.0, 4.0, Rintegral24, avgBesselmatrix2, avgBesselmatrix4 ),(4.0, 4.0, Rintegral44, avgBesselmatrix4, avgBesselmatrix4))
        
        F_queue = Queue()
        F_processes1 = [Process(target=FirstSecond, args=(F_queue, z[0], z[1])) for z in zip(range(6),F_inputs)]
        F_processes2 = [Process(target=FirstSecond2, args=(F_queue, z[0], z[1])) for z in zip(range(6,12),F_inputs)]
        F_processes = F_processes1 + F_processes2
        for pF in F_processes:
            pF.start()
        
        Ts = [F_queue.get() for pF in F_processes]
        Ts.sort()
        Total = [T[1] for T in Ts]

        self.covariance00 = np.vstack((Total[0], Total[6]))
        self.covariance02 = np.vstack((Total[1], Total[7]))
        self.covariance04 = np.vstack((Total[2], Total[8]))
        self.covariance22 = np.vstack((Total[3], Total[9]))
        self.covariance24 = np.vstack((Total[4], Total[10]))
        self.covariance44 = np.vstack((Total[5], Total[11]))

        print 'RSD_shell_covariance_AllModes is finished'
        return self.covariance00, self.covariance02, self.covariance04, self.covariance22, self.covariance24, self.covariance44






def Ll(l,x):
    
    """ Legendre Polynomial
        call fortran module in legen.f90 """
    
    import numpy as np
    from numpy import vectorize
    from fortranfunction import eval_legendre
    
    Le_func = lambda lp,xp : eval_legendre(lp,xp)
    Le_func = np.vectorize(Le_func)
    result = Le_func(l,x)
    
    return result




def get_closest_index_in_data( value, data ):

    for i in range(len(data)):
        if data[i] < value : pass
        elif data[i] >= value :
            if np.fabs(value - data[i]) > np.fabs(value - data[i-1]):
                value_index = i-1
            else : value_index = i
            break
    
    return value_index


def avgBessel(l,k,rmin,rmax):
    
    """ Averaged spherical Bessel function in configuration space
        call fortran module in sici.f90 (sine integral ftn) """
    
    from numpy import vectorize, pi, cos, sin
    from fortranfunction import sici
    sici = vectorize(sici)
    
    if l == 0.0 :
        result = 4. * pi * (-k * rmax * cos(k * rmax) + k * rmin * cos(k * rmin) + sin(k * rmax) - sin(k * rmin))/(k**3)
    elif l == 2.0 :
        result = 4. * pi * (k * rmax * cos(k * rmax) - k*rmin*cos(k*rmin)-4*sin(k*rmax) +
                          4*sin(k*rmin) + 3*sici(k * rmax) - 3*sici(k*rmin))/k**3
    else :
        result = (2.* pi/k**5) * ((105 * k/rmax - 2 * k**3 * rmax) * cos(k * rmax) +\
                                  (- 105 * k/rmin + 2 * k**3 * rmin) * cos(k * rmin) +\
                                  22 * k**2 * sin(k *rmax) - (105 * sin(k * rmax))/rmax**2 -\
                                  22 * k**2 *sin(k * rmin) + (105 * sin(k * rmin))/rmin**2 +\
                                  15 * k**2 * (sici(k * rmax) - sici(k * rmin)))
    return result



def symmetrize(cov):
    
    for i in range(len(cov.diagonal())):
        for j in range(len(cov.diagonal())):
            cov[j,i] = cov[i,j]
    return cov


def CombineMatrix3by3(cov00, cov01, cov02, cov10, cov11, cov12, cov20, cov21, cov22):
    C_Matrix = np.array([[cov00,cov01,cov02],\
                        [cov10,cov11,cov12],\
                        [cov20,cov21,cov22]])
    return C_Matrix

def CombineMatrix2by2(cov00, cov01, cov10, cov11):
        #
        #   Input should be matrix
        #   matrices = [00, 02, 22]
        #
    C_Matrix = np.array([[cov00,cov01],\
                        [cov10,cov11]])
    return C_Matrix

def CombineMatrix3by2(cov00, cov01, cov10, cov11, cov20, cov21):
    C_Matrix = np.array([[cov00,cov01],\
                        [cov10,cov11],\
                        [cov20,cov21]])

    return C_Matrix

def CombineCovariance3(l, matrices):

    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 22, 24, 44] """

    cov00 = matrices[0][0:l+1,0:l+1]
    cov02 = matrices[1][0:l+1,0:l+1]
    cov04 = matrices[2][0:l+1,0:l+1]
    cov20 = matrices[3][0:l+1,0:l+1]
    cov22 = matrices[4][0:l+1,0:l+1]
    cov24 = matrices[5][0:l+1,0:l+1]
    cov40 = matrices[6][0:l+1,0:l+1]
    cov42 = matrices[7][0:l+1,0:l+1]
    cov44 = matrices[8][0:l+1,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))

    return C_Matrix


def CombineCrossCovariance3(l1, l2, matrices):

    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
        
    cov00 = matrices[0][0:l1+1,0:l2+1]
    cov02 = matrices[1][0:l1+1,0:l2+1]
    cov04 = matrices[2][0:l1+1,0:l2+1]
    cov20 = matrices[3][0:l1+1,0:l2+1]
    cov22 = matrices[4][0:l1+1,0:l2+1]
    cov24 = matrices[5][0:l1+1,0:l2+1]
    cov40 = matrices[6][0:l1+1,0:l2+1]
    cov42 = matrices[7][0:l1+1,0:l2+1]
    cov44 = matrices[8][0:l1+1,0:l2+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))

    return C_Matrix


def CombineCovariance2(l, matrices):
    
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][0:l+1,0:l+1]
    cov02 = matrices[1][0:l+1,0:l+1]
    cov20 = matrices[3][0:l+1,0:l+1]
    cov22 = matrices[4][0:l+1,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))

    return C_Matrix


def CombineCrossCovariance2(l1, l2, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][0:l1+1,0:l2+1]
    cov02 = matrices[1][0:l1+1,0:l2+1]
    cov20 = matrices[3][0:l1+1,0:l2+1]
    cov22 = matrices[4][0:l1+1,0:l2+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    C_Matrix = np.vstack((C_Matrix1, C_Matrix2))
    
    return C_Matrix


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
    
    Matrix1 = np.concatenate((dxib0, dxib2, dxib4), axis=1)
    Matrix2 = np.concatenate((dxif0, dxif2, dxif4), axis=1)
    Matrix3 = np.concatenate((dxis0, dxis2, dxis4), axis=1)
    Xi = np.vstack((Matrix1, Matrix2, Matrix3))

    Matrix1 = np.concatenate((dxib0, dxib2), axis=1)
    Matrix2 = np.concatenate((dxif0, dxif2), axis=1)
    Matrix3 = np.concatenate((dxis0, dxis2), axis=1)
    Xi2 = np.vstack((Matrix1, Matrix2, Matrix3))
    
    return Xi, Xi2

def CombineDevXi3(l, matrices):
    """ Input should be a list of matrices :
        matrices = [00, 02, 04, 20, 22, 24, 40, 42, 44] """
    
    cov00 = matrices[0][:,0:l+1]
    cov02 = matrices[1][:,0:l+1]
    cov04 = matrices[2][:,0:l+1]
    cov20 = matrices[3][:,0:l+1]
    cov22 = matrices[4][:,0:l+1]
    cov24 = matrices[5][:,0:l+1]
    cov40 = matrices[6][:,0:l+1]
    cov42 = matrices[7][:,0:l+1]
    cov44 = matrices[8][:,0:l+1]
    
    C_Matrix1 = np.concatenate((cov00, cov02, cov04), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22, cov24), axis=1)
    C_Matrix3 = np.concatenate((cov40, cov42, cov44), axis=1)
    Xi = np.vstack((C_Matrix1, C_Matrix2, C_Matrix3))
    
    C_Matrix1 = np.concatenate((cov00, cov02), axis=1)
    C_Matrix2 = np.concatenate((cov20, cov22), axis=1)
    Xi2 = np.vstack((C_Matrix1, C_Matrix2))
    
    return Xi, Xi2


def FisherProjection( deriv, CovMatrix ):
    
    """ Projection for Fisher Matrix """
    inverseC = inv(CovMatrix)
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

def FractionalError( param1, param2, CovarianceMatrix  ):
    
    """ marginalized cov : fractional error on Parameter  \sigma P / P """
        
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error[0]/param1, error[1]/param2


def Local_FractionalError( param1, FisherMatrix  ):
    
    local_error = 1./np.sqrt(np.diagonal(FisherMatrix))/param1
    """ fractional error on Parameter  \sigma P / P """

    return local_error


def FractionalErrorBand( params, CovarianceMatrix  ):
    
    """ fractional error on Parameter  \sigma P / P """
    
    error = np.sqrt(CovarianceMatrix.diagonal())
    return error/params


def CrossCoeff( Matrix ):
    
    """ Cross Corelation matrix   C_ij / Sqrt( C_ii * C_jj) """
    
    matrix1,matrix2 = np.mgrid[0:len(Matrix[0]),0:len(Matrix[0])]
    diagonal = Matrix.diagonal()
    Coeff = Matrix /np.sqrt(diagonal[matrix1] * diagonal[matrix2])
    return Coeff

def cumulative_SNR( data_Vec, Cov ):

    cumul_SNR = []
    for i in range(len(data_Vec)):
        InvCov = np.linalg.inv(Cov[0:i+1, 0:i+1])
        SNR = np.dot( np.dot( data_Vec[0:i+1], InvCov ), data_Vec[0:i+1])
        cumul_SNR.append(SNR)
        #print np.shape(data_Vec[0:i+1]),np.shape(InvCov)
    return np.array(cumul_SNR)
                     


def Linear_plot( base, valuename, *args, **kwargs ):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    basename = kwargs.get('basename','k')
    title = kwargs.get('title', 'Fractional Error')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    xmin = kwargs.get('xmin',10**(-4))
    xmax = kwargs.get('xmax', 1000.)
    ymin = kwargs.get('ymin', 10**(-7))
    ymax = kwargs.get('ymax', 10**(5))
    scale = kwargs.get('scale', None )
    ylabel = kwargs.get('ylabel', 'Fractional Error')
    
    #linestyles = ['b-', 'r.', 'g^','c.', 'm--', 'y.', 'k.']
    linestyles = ['k-', 'r^', 'r^', 'g.','m-', 'g-', 'y--', 'k--','ro','co', 'mo', 'yo', 'ko']
    ziplist = zip(args, valuename, linestyles)
    
    fig = plt.figure()
    fig.suptitle( title , fontsize=10 )
    
    if scale == None:
        for z in ziplist: plt.plot( base, z[0], z[2], label = z[1] )
    elif scale == 'log':
        for z in ziplist: plt.loglog( base, z[0], z[2], label = z[1] )
    elif scale == 'semilogy':
        for z in ziplist: plt.semilogy( base, z[0], z[2], label = z[1] )
    elif scale == 'semilogx':
        for z in ziplist: plt.semilogx( base, z[0], z[2], label = z[1] )

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel( basename )
    plt.ylabel(ylabel)
    plt.legend(loc=4,prop={'size':10})
    plt.grid(True)
    #plt.show()
    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    #plt.clf()
    print "pdf file saved : ", pdfname

def Linear_plot2( base, base2, value2, valuename, *args,**kwargs):
    #
    #
    #
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    
    basename = kwargs.get('basename','k')
    title = kwargs.get('title', 'Fractional Error')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    xmin = kwargs.get('xmin',10**(-4))
    xmax = kwargs.get('xmax', 1000.)
    ymin = kwargs.get('ymin', 10**(-7))
    ymax = kwargs.get('ymax', 10**(5))
    scale = kwargs.get('scale', None )
    
    #linestyles = ['b-', 'r.', 'g^','c.', 'm--', 'y.', 'k.']
    linestyles = ['b--', 'r^', 'g.','c--', 'm--', 'y--', 'k--','ro','co', 'mo', 'yo', 'ko']
    ziplist = zip(args, valuename, linestyles)
    
    fig = plt.figure()
    fig.suptitle( title , fontsize=10 )
    
    if scale == None:
        for z in ziplist: plt.semilogx( base, z[0], z[2], label = z[1] )
    elif scale == 'log':
        for z in ziplist: plt.loglog( base, z[0], z[2], label = z[1] )
        plt.loglog(base2, value2, 'b-')
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel( basename )
    plt.ylabel('Fractional Error')
    plt.legend(loc=3,prop={'size':10})
    plt.grid(True)
    #plt.show()
    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    #plt.clf()
    print " pdf file saved : ", pdfname

def Contour_plot( base, crosscoeffdata, **kwargs ):
    #
    #   Make 2-D Contour Plot for Covariance Matrix and Fisher Matrix
    #
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    
    basename = kwargs.get('basename','log10(k)')
    title = kwargs.get('title', 'Covariance')
    pdfname = kwargs.get('pdfname', 'test.pdf')
    scale = kwargs.get('scale', None )
    
    k1 = np.log10(base)
    k2 = np.log10(base)
    
    fig, ax = plt.subplots()
    ax.set_title( title )
    ax.set_xlabel( basename )
    ax.set_ylabel( basename )
    
    if scale == None:
        data = crosscoeffdata
        label = 'Amplitude'
    elif scale == 'log':
        data = np.log10(crosscoeffdata)
        label = '$\log_{10}$(Amplitude)'
    elif scale == 'asinh':
        data = np.arcsinh(1.0 * crosscoeffdata)/1.0
        label = 'Amplitude'
    
    cax = ax.imshow(data, extent=(k1.min(), k1.max(), k2.max(), k2.min()), interpolation='nearest', cmap=cm.gist_rainbow)
                    
    cbar = fig.colorbar(cax, ax = ax )
    cbar.set_label( label )
                    
    pdf=PdfPages( pdfname )
    pdf.savefig(fig)
    pdf.close()
    print " pdf file saved : ", pdfname

def makedirectory(dirname):
    import os
    if not os.path.exists("./" + dirname + "/"):
        os.mkdir("./" + dirname + "/")
