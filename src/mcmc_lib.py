import emcee
import os
import numpy as np
import sys
import copy
import ctypes
import itertools
from run_error_analysis import *


def compute_datavector(lik_class):

    datavP = np.zeros( lik_class.kcenter_y.size * 3 )
    datavXi = np.zeros( lik_class.rcenter.size * 3 )

    if 'p' in lik_class.probe : 
        datavP = P_multipole(None, lik_class)
    if 'xi' in lik_class.probe :
        #datavP = P_multipole(None, lik_class)
        if 'p' not in lik_class.probe:
            lik_class.multipole_P_interp(0)
            lik_class.multipole_P_interp(2)
            lik_class.multipole_P_interp(4)

        else : 
            datavP = P_multipole(None, lik_class)

        datavXi = Xi_multipole(None, lik_class)
    datav = np.hstack([datavP, datavXi])
    return datav
        
def log_like_wrapper(lik_class):
    
    datav_fid_ = lik_class.datav_fid   
    fisher = lik_class.fisher
    mask = lik_class.mask
    datav_ = compute_datavector(lik_class)
    
    #print 'fisher', fisher.shape, 'mask', np.sum(mask), 'datav', datav_.shape, 'datavfid', datav_fid_.shape

    datav_fid = datav_fid_[mask]
    datav = datav_[mask]

    datavdiff = np.fabs(datav - datav_fid)
    chisqr = np.dot(np.dot(datavdiff, fisher), datavdiff.T)
     
    return -0.5 * chisqr


    
class LikelihoodFunctionWrapper(object):
    def __init__(self, lik_class, varied_parameters, cosmo_names, cosmo_min, cosmo_fiducial, cosmo_max):
        self.varied_parameters = varied_parameters
        self.cosmo_names = cosmo_names
        self.cosmo_min = cosmo_min
        self.cosmo_fid = cosmo_fiducial
        self.cosmo_max = cosmo_max
        self.lik_class = lik_class

    def fill_varied(self, x):

        assert len(x) == len(self.varied_parameters), "Wrong number of parameters"
        
        filter = np.zeros(len(self.cosmo_names))
        for i, var in enumerate(self.cosmo_names):
            if var in self.varied_parameters:
                filter[i] = x[i]
            else : filter[i] = self.cosmo_fid[i]                
        
        self.lik_class.b = filter[0]
        self.lik_class.f = filter[1]
        self.lik_class.s = filter[2]
        self.lik_class.nn = filter[3]
        
        
    def prior_cosmology(self):
        good = True
        for i, v in enumerate(self.cosmo_fid):
            min_v = self.cosmo_min[i]
            max_v = self.cosmo_max[i]
            if v<min_v or v>max_v: good=False
        if good:
            return 0.0
        else:
            return -np.inf


    def __call__(self, x):
        #icp = copy.deepcopy(self.cosmo_fid)
        self.fill_varied(x)
        flat_prior = self.prior_cosmology()

        if not np.isfinite(flat_prior):
            #print "outside flat prior range"
            return -np.inf
        else : like = log_like_wrapper(self.lik_class)
        print 'filled value', self.lik_class.b, self.lik_class.f, self.lik_class.s, self.lik_class.nn, 'like =', like
        return like



        
global my_likelihood


def likelihood_task(p):
    return my_likelihood(p)

        
def sample_main(lik_class, varied_parameters, iterations, nwalker, nthreads, 
        filename, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, pool=None):

    likelihood = LikelihoodFunctionWrapper(lik_class, varied_parameters, 
        cosmo_names, cosmo_min, cosmo_fid, cosmo_max)
    global my_likelihood
    my_likelihood = likelihood
    if pool is not None:
        if not pool.is_master():
            print "Slave core waiting for tasks now"
            pool.wait()
            return

    #starting_point = []
    #starting_point += cosmo_fid.convert_to_vector_filter(varied_parameters)
    filter = np.zeros(len(cosmo_names), dtype=bool)
    for i, val in enumerate(cosmo_names) :
        if val in varied_parameters: filter[i] = 1

    starting_point = cosmo_fid[filter]  
    print "Starting point = ", starting_point

    std = cosmo_fid_sigma[filter]
    p0 = emcee.utils.sample_ball(starting_point, std, size=nwalker)

    ndim = len(starting_point)
    print "ndim = ", ndim
    print "start = ", starting_point
    print "std = ", std
    
    sampler = emcee.EnsembleSampler(nwalker, ndim, likelihood_task, threads=nthreads, pool=pool)

    
    import time, sys
    i = 0
    t0 = time.time() 
    """
    p, loglike, state =  sampler.run_mcmc(p0, iterations)
    samples = sampler.flatchain
    header =  '# ' + '    '.join(varied_parameters)+" \n"
    np.savetxt(filename, samples, header=header)
   
    """
    f = open(filename, 'w')
    #write header here
    f.write('# ' + '    '.join(varied_parameters)+"  log_like\n")
    
    for i, result in enumerate(sampler.sample(p0, iterations=iterations, storechain=False)):
        for row,logl in zip(result[0],result[1]):
            #if (i+1) % 100 == 0:
            #    print("{0:5.1%}".format(float(i) / iterations))
            p_text = '  '.join(str(r) for r in row)
            #print '%s %e\n' % (p_text, logl)
            f.write('%s %e\n' % (p_text, logl))

        if i == 0: 
            t1 = time.time() - t0 
            totTime = t1 * iterations * 0.5
        sys.stdout.write('iteration {}/{} remained time {} s \r'.format(i+1, iterations, (iterations-i)*t1))
        sys.stdout.flush()
        i+=1

        f.flush()
    f.close()
    
    print 'elapsed time ', time.time() - t0, ' s'
    print "save chain to", filename, '\n'
    


def snake_sampler_main(lik_class, varied_parameters, iterations, nwalker, nthreads, 
        filename, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
        nsample_dimension, threshold, maxiter, pool=None):
    


    likelihood = LikelihoodFunctionWrapper(lik_class, varied_parameters, 
        cosmo_names, cosmo_min, cosmo_fid, cosmo_max)
    global my_likelihood
    my_likelihood = likelihood
    #if pool is not None:
    #    if not pool.is_master():
    #        print "Slave core waiting for tasks now"
    #        pool.wait()
    #        return

    #starting_point = []
    #starting_point += cosmo_fid.convert_to_vector_filter(varied_parameters)
    filter = np.zeros(len(cosmo_names), dtype=bool)
    for i, val in enumerate(cosmo_names) :
        if val in varied_parameters: filter[i] = 1

    starting_point = cosmo_fid[filter]  
    print " Starting point = ", starting_point
    ndim = len(starting_point)

    #nsample_dimension = 10
    #threshold = 4 
    grid_size = 1.0/nsample_dimension
    origin = starting_point #self.pipeline.normalize_vector(self.pipeline.start_vector())
    spacing = np.repeat(grid_size, ndim)



    from snake import Snake
    snake = Snake(likelihood_task, origin, spacing, threshold, pool=pool)
    X,P,E = snake.iterate()
    #while not snake.converged():

    return 0


    import time, sys
    i = 0
    t0 = time.time() 
    f = open(filename, 'w')
    #write header here
    f.write('# ' + '    '.join(varied_parameters)+"  log_like\n")

    
    for (params,post,extra) in zip(X,P,E):
        #print params, post, extra
        p_text = '  '.join(str(x) for x in params)
        print i, ' %s %e\n' % (p_text, post)

        #sys.stdout.flush()
        f.write('%s %e\n' % (p_text, post))
        f.flush()

        #if i == 0: 
        #    t1 = time.time() - t0 
        #    totTime = t1 * maxiter * 0.5
        #sys.stdout.write('iteration {}/{} remained time {} s \r'.format(i+1, maxiter, (maxiter-i)*t1))
        #sys.stdout.flush()
        i+=1
    f.close()
    sys.stderr.write("Ran for %d iterations\n"%snake.iterations)
    

def grid_sampler_main(lik_class, varied_parameters, iterations, nwalker, nthreads, 
        filename, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
        nsample, nstep, maxiter, pool=None):

    import time, sys
    i = 0
    t0 = time.time() 

    likelihood = LikelihoodFunctionWrapper(lik_class, varied_parameters, 
        cosmo_names, cosmo_min, cosmo_fid, cosmo_max)
    global my_likelihood
    my_likelihood = likelihood
    #if pool is not None:
    #    if not pool.is_master():
    #        print "Slave core waiting for tasks now"
    #        pool.wait()
    #        return

    #starting_point = []
    #starting_point += cosmo_fid.convert_to_vector_filter(varied_parameters)
    filter = np.zeros(len(cosmo_names), dtype=bool)
    varied_cosmo_names = []
    for i, val in enumerate(cosmo_names) :
        if val in varied_parameters: 
            filter[i] = 1
            varied_cosmo_names.append(cosmo_names[i])

    starting_point = cosmo_fid[filter]  
    print " Starting point = ", starting_point
    ndim = len(starting_point)


    #nsample = 100
    #nstep = 10
    #filename = ""
    from grid_sampler import GridSampler
    GS = GridSampler(pool=pool)
    GS.config( varied_parameters, cosmo_names, cosmo_min[filter], 
        cosmo_fid[filter], cosmo_max[filter], nsample, None, filename )
    #GS.setup_sampling()

    if GS.sample_points is None:
        GS.setup_sampling()

    #Chunk of tasks to do this run through, of size nstep.
    #This advances the self.sample_points forward so it knows
    #that these samples have been done
    samples = list(itertools.islice(GS.sample_points, GS.nsample**len(GS.varied_params)))
    #samples = list(itertools.islice(GS.sample_points, 0, GS.nstep))

    #If there are no samples left then we are done.
    if not samples:
        GS.converged=True
        return

    #Each job has an index number in case we are saving
    #the output results from each one
    #sample_index = np.arange(len(samples)) + GS.ndone
    #jobs = list(zip(sample_index, samples))
    #jobs = list(zip(samples, sample_index))
    jobs = list(samples)

    #Actually compute the likelihood results
    if GS.pool:
        results = GS.pool.map(likelihood_task, jobs)
    else:
        results = list(map(likelihood_task, jobs))

    if results == None: return 0
    else : pass

    #Update the count
    #GS.ndone += len(results)
    #Save the results of the sampling

    f = open(filename, 'w')
    f.write('#')
    f.write(''.join('cosmological_parameter--'+param+'  ' for param in varied_cosmo_names)+'  post\n' )
    f.write('#n_varied='+str(len(varied_parameters))+'\n' )
    f.write('#sampler=grid\n' )
    f.write('#nsample_dimension='+str(nsample)+'\n')
    f.write('#nstep=-1\n')
    #f.write('#' + '    '.join(varied_parameters)+"  log_like\n")
    for sample, post in zip(samples, results):
        #Optionally save all the results calculated by each
        #pipeline run to files
        #print (sample, result)
        
        #(prob, extra) = result
        #always save the usual text output
        #print (sample, extra, prob)
        #self.output.parameters(sample, extra, prob)
        #p_text = '  '.join(str(x) for x in sample)
        p_text = ''.join('{:10.10f}  '.format(x) for x in sample)
        #print i, ' %s %e\n' % (p_text, post)
        f.write('%s  %10.10f\n' % (p_text, post))
        #f.write('%10.10f %10.10f\n' % (sample[0], post))
        f.flush()

    print 'elapsed time ', time.time() - t0, ' s'
    print "save chain to", filename, '\n'

    return 0
    #f.close()
  

def chisquare_fmin( X, lik_class, Nfeval):
    

    ndim = len(X)
    lik_class.b = X[0]

    if ndim == 2 :
        lik_class.f = X[1]*0.74/2.
    elif ndim == 3 :
        lik_class.f = X[1]*0.74/2.
        lik_class.s = X[2] * 3.5/2.
    elif ndim == 4 :
        lik_class.f = X[1]*0.74/2.
        lik_class.s = X[2] * 3.5/2.
        lik_class.nn= X[3]*0.0003/2.

    like = log_like_wrapper(lik_class)
    chi2 = -2. * like

    print ' {0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.\
    format(Nfeval, lik_class.b, lik_class.f, lik_class.s, lik_class.nn, chi2 )
    Nfeval += 1

    return chi2


def fmin_sampler_main(lik_class, varied_parameters, iterations, nwalker, nthreads, 
        filename, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, cosmo_fid_sigma, 
        nsample, nstep, maxiter, pool=None):

    import time, sys
    i = 0
    t0 = time.time() 
    """
    likelihood = LikelihoodFunctionWrapper(lik_class, varied_parameters, 
        cosmo_names, cosmo_min, cosmo_fid, cosmo_max)
    global my_likelihood
    my_likelihood = likelihood
    """
    #if pool is not None:
    #    if not pool.is_master():
    #        print "Slave core waiting for tasks now"
    #        pool.wait()
    #        return

    #starting_point = []
    #starting_point += cosmo_fid.convert_to_vector_filter(varied_parameters)
    filter = np.zeros(len(cosmo_names), dtype=bool)
    varied_cosmo_names = []
    for i, val in enumerate(cosmo_names) :
        if val in varied_parameters: 
            filter[i] = 1
            varied_cosmo_names.append(cosmo_names[i])

    rescale_factor = np.array([1, 2/0.74, 2/3.5, 2/0.0003])[filter]
    #cosmo_fid_rescaled = cosmo_fid * rescale_factor
    #starting_point_rescaled = cosmo_fid_rescaled[filter] 
    starting_point =  cosmo_fid[filter] 
    print " Starting point = ", starting_point
    ndim = len(starting_point)

    from scipy import optimize
    #global Nfeval 
    Nfeval = 1
    print  ' {0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}'.format('Iter', ' b', ' f', ' s', ' nn', ' chi2')  

    """
    minimum = optimize.fmin(chisquare_fmin, list(starting_point * rescale_factor), args=(lik_class, Nfeval),
                            xtol = 0.001, full_output=True, retall=True )

    bestfit = minimum[0]* 1./rescale_factor
    min_chi2 = minimum[1]
    iteration = minimum[2]
    """

    minimum = optimize.minimize(chisquare_fmin, list(starting_point * rescale_factor), args=(lik_class, Nfeval),
                       method='Nelder-Mead', #tol=0.0001, 
                       options={'xatol': 0.01, 'fatol': 0.01, 'return_all': True, 'adaptive': True, 'disp': True}
                       )
    bestfit = minimum.x * 1./rescale_factor #minimum[0]
    min_chi2 = minimum.fun
    iteration = minimum.nit 
    

    p_text = ''.join('{:10.10f}  '.format(x) for x in bestfit)
    print 'minimum=', p_text
       

    f = open(filename, 'w')
    f.write('#fmin minimization\n')
    f.write('#n_varied='+str(len(varied_parameters))+'\n' )
    f.write('#iteration='+str(iteration)+'\n')
    #f.write('#parameters :\n')
    #f.write(''.join('#      cosmological_parameter--'+param+'='+bv+'\n' for (param, bv) in zip(varied_parameters, cosmo_fid)) )
    f.write('#bestfit : \n#')
    #f.write('#chisquqre='+str(min_chi2)+'\n')
    #f.write(''.join('#  cosmological_parameter--'+param+'='+bv'\n' for (param, bv) in zip(varied_cosmo_names, bestfit) +'chi2' )
    f.write(''.join('cosmological_parameter--'+param+'  ' for param in varied_cosmo_names)+'  chi2\n' )
    p_text = ''.join('{:10.10f}  '.format(x) for x in bestfit)
    f.write('%s  %10.10f\n' % (p_text, min_chi2))
    #f.write('#')
    #f.write(''.join('cosmological_parameter--'+param+'  ' for param in varied_cosmo_names)+'  chi2\n' )
    f.flush()
    f.close()

    print "save chain to", filename
    print 'elapsed time ', time.time() - t0, ' s' 

    return 0
