import emcee
import os
import numpy as np
import sys
import copy
import ctypes

from run_error_analysis import *


def compute_datavector(lik_class):

    datavP = P_multipole(None, lik_class)
    datavXi = Xi_multipole(None, lik_class)
    datav = np.hstack([datavP, datavXi])
    return datav
        
def log_like_wrapper(lik_class):
    
    datav_fid = lik_class.datav_fid   
    fisher = lik_class.fisher
    mask = lik_class.mask
    datav = compute_datavector(lik_class)
    
    #print 'fisher', fisher.shape, 'mask', np.sum(mask), 'datav', datav.shape

    datav_fid = datav_fid[mask]
    datav = datav[mask]
    
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
            
        #print 'filled value', self.lik_class.b, self.lik_class.f, self.lik_class.s, self.lik_class.nn, 'like =', like
        return like



        
global my_likelihood


def likelihood_task(p):
    return my_likelihood(p)

        
def _sample_main(lik_class, varied_parameters, iterations, nwalker, nthreads, 
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
    print " Starting point = ", starting_point
    ndim = len(starting_point)

    #nsample_dimension = 10
    #threshold = 4 
    grid_size = 1.0/nsample_dimension
    #maxiter = 100000 
    print " nsample_dimension:", nsample_dimension
    print " threshold:", threshold
    print " maxiter:", maxiter
    
    origin = starting_point #self.pipeline.normalize_vector(self.pipeline.start_vector())
    spacing = np.repeat(grid_size, ndim)

    from snake import Snake
    snake = Snake(likelihood_task, origin, spacing, threshold, pool=pool)
    while not snake.converged():
        snake.iterate()

    for p, L in list(snake.likelihoods.items()):
        print(p[0], p[1], L)
    sys.stderr.write("Ran for %d iterations\n"%snake.iterations)
