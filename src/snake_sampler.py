from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
#from .. import ParallelSampler
import numpy as np
from snake import Snake

def _posterior(p_in):
    #Check the normalization
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
        print(p_in)
        return -np.inf, [np.nan for i in range(len(snake_pipeline.extra_saves))]
    p = snake_pipeline.denormalize_vector(p_in)
    like, extra = snake_pipeline.posterior(p)
    return like, extra


class ParallelSampler():
    parallel_output = True
    is_parallel_sampler = True
    supports_smp = True
    #def __init__(self, ini, pipeline, output, pool=None):
    def __init__(self, pool=None):
        #Sampler.__init__(self, ini, pipeline, output)
        self.pool = pool

    def worker(self):
        ''' Default to a map-style worker '''
        if self.pool:
            self.pool.wait()
        else:
            raise RuntimeError("Worker function called when no parallel pool exists!")

    def is_master(self):
        return self.pool is None or self.pool.is_master()



class SnakeSampler(ParallelSampler):
    sampler_outputs = [("post", float)]
    parallel_output = False

    def config(self, posterior, varied_params, cosmo_names, cosmo_min, cosmo_fid, cosmo_max, nsample_dimension, threshold, maxiter, filename):
        #global snake_pipeline
        #snake_pipeline=self.pipeline
        #if self.is_master():
        #threshold = threshold #self.read_ini("threshold", float, 4.0)
        self.grid_size = 1.0/nsample_dimension #self.read_ini("nsample_dimension", int, 10)
        self.maxiter = maxiter #self.read_ini("maxiter", int, 100000)
        self.varied_params = varied_params #[0,1,2]
        self.params_fid = cosmo_fid
        self.params_min = cosmo_min
        self.params_max = cosmo_max
        #self.posterior = posterior
        threshold = threshold
        origin = self.params_fid
        spacing = np.repeat(self.grid_size, len(self.varied_params))
        #origin = self.pipeline.normalize_vector(self.pipeline.start_vector())
        #spacing = np.repeat(self.grid_size, len(self.pipeline.varied_params))
        self.snake = Snake(posterior, origin, spacing, threshold, pool=self.pool)


    def execute(self):

        #self.snake = Snake(self.posterior, self.origin, self.spacing, self.threshold, pool=self.pool)
        X, P, E = self.snake.iterate()
        for (x,post,extra) in zip(X,P,E):
            try:
                x = self.varied_params #self.pipeline.denormalize_vector(x)
                print (x,post)
                #self.output.parameters(x, extra, post)
            except ValueError:
                print("The snake is trying to escape its bounds!")



    def is_converged(self):
        if self.snake.converged():
            print("Snake has converged!")
            print("Best post = %f    Best surface point = %f" %(self.snake.best_like_ever, self.snake.best_fit_like))
            return True
        if self.snake.iterations > self.maxiter:
            print("Run out of iterations.")
            print("Done %d, max allowed %d" % (self.snake.iterations, self.maxiter))
            return True
        return False
