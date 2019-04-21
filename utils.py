import numpy as np
from numpy import zeros, sqrt, pi, vectorize
from numpy.linalg import pinv, inv
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


def plot_fisher_corr_matrix(fisher, labels = [r'$b$', r'$f$', r'$\sigma_{v,p}$', r'$1/\bar{n}$',r'$b$', r'$f$', r'$\sigma_{v,\xi}$']):    
    
    
    nx, ny = fisher.shape
    cov = inv(fisher)
    corrmat = correlation_matrix(cov)
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (14,6))
    im = ax.imshow(fisher, interpolation = 'none')
    im2 = ax2.imshow(corrmat, interpolation = 'none')
    fig.colorbar(im, ax=ax)
    fig.colorbar(im2, ax=ax2)

    ax.set_xticks(np.arange(0, nx, 1));
    ax.set_yticks(np.arange(0, nx, 1));
    ax.set_xticklabels(labels);
    ax.set_yticklabels(labels);
    ax.set_title('Fisher')
    ax2.set_xticks(np.arange(0, nx, 1));
    ax2.set_yticks(np.arange(0, nx, 1));
    ax2.set_xticklabels(labels);
    ax2.set_yticklabels(labels);
    ax2.set_title('corr')
    
def correlation_matrix(mat):
    
    nx, ny = mat.shape
    corrmat = np.zeros((nx, ny))
    for i in np.arange(nx):
        for j in np.arange(ny):
            corrmat[i][j] = mat[i][j] / np.sqrt(mat[i][i]*mat[j][j] )
    return corrmat

def printout_matrix_component(mat):
    nx, ny = mat.shape
    print ''
    start = ''
    for i in np.arange(nx):
        for j in np.arange(ny):
            start+='  {:3.3e}'.format(mat[i][j])

        print '['+ start+']'
        start = ''     
        
        
        