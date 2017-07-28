import sys, os
import matplotlib.pyplot as plt
from noshellavg import *



def plot_single_ellipse(fisher_params_filename, parameter_name = [r'$b$', r'$f$', r'$\sigma_v$'],
                 parameter_truth = [2.0, 0.74, 3.5], keep = [0,1], 
                 extent =[[0.975, 1.025], [0.85, 1.15], [0.5, 1.5]],
                 text_label = '\n', out_name = 'figure/ellipse_test.png', diffsky=False, dss=False):

    from numpy.linalg import inv
    
    fig, ax = plt.subplots(1,1, figsize = (6,5))
    
    
    truth_x, truth_y = parameter_truth[keep[0]], parameter_truth[keep[1]]
    label_x, label_y = parameter_name[keep[0]], parameter_name[keep[1]]

    extent_x = extent[keep[0]]
    extent_y = extent[keep[1]]
        
    Covlist_mar = []
        
    for i in range(len(fisher_params_filename)):
        data = np.loadtxt(fisher_params_filename[i])  

        row_x, row_y = data.shape
        N_params = int(np.sqrt(row_x))

        mask1 = np.zeros((N_params,N_params), dtype=bool)
        mask2 = np.zeros((N_params,N_params), dtype=bool)
        mask1[:,keep] = 1
        mask2[keep,:] = 1
        mask = mask1*mask2

        if dss == 1 : 
            mask_dss_p = np.ones((N_params,N_params), dtype=bool)
            mask_dss_p[:,3] = 0
            mask_dss_p[3,:] = 0

        
            mask_dss_xi = np.ones((N_params,N_params), dtype=bool)
            mask_dss_xi[:,2] = 0
            mask_dss_xi[2,:] = 0

            Fisherlist = [ data[:,q].reshape(N_params,N_params) for q in range(1,row_y)]
            
            
            CovP = inv(Fisherlist[0][mask_dss_p].reshape(3,3))
            CovXi = inv(Fisherlist[1][mask_dss_xi].reshape(3,3))
            Covtot = inv(Fisherlist[2])
            
            Covlist = [CovP, CovXi, Covtot]
                         

            
        else : Covlist = [ inv(data[:,q].reshape(N_params,N_params)) for q in range(1,row_y)]
        
        
        if diffsky: 
            C_diffsky = inv((data[:,1]+data[:,2]).reshape(N_params,N_params))
            Covlist += [C_diffsky]
            
            print data[:,3]
            print data[:,1]+data[:,2]
        Covlist_mar += [cov[mask].reshape(2,2) for cov in Covlist]
        
    
    el = confidence_ellipse(truth_x,truth_y, None, None, *tuple(Covlist_mar) )
    for e in el:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)

    xmin = truth_x*extent_x[0]
    xmax = truth_x*extent_x[1]
    ymin = truth_y*extent_y[0]
    ymax = truth_y*extent_y[1]

    ax.tick_params(labelsize=15)
    ax.locator_params(axis = 'x', nbins=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(label_y, size = 20)
    ax.set_xlabel(label_x, size = 20)
    labellist = ['P', 'Xi', 'tot']
    ax.text(xmin * 1.0005, ymin, text_label, size = 15) 
    fig.tight_layout()
    fig.savefig(out_name)
    
    
def plot_triple_ellipse(fisher_params_filename, parameter_name = [r'$b$', r'$f$', r'$\sigma_v$'],
                        parameter_truth = [2.0, 0.74, 3.5], 
                        extent =[[0.975, 1.025], [0.85, 1.15], [0.5, 1.5]],
                        linecolor = ['b', 'r', 'g', 'b', 'r', 'g', 'y', 'c', 'k'],
                        linestyle = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid'],
                        text_label = '\n', out_name = 'figure/ellipse_test.png', diffsky=0, dss = 0):

    from numpy.linalg import inv
    
    #data = np.loadtxt(fisher_params_filename)
    #row_x, row_y = data.shape
    #N_params = int(np.sqrt(row_x))
    keeps = [[0,1], [0,2], [1,2]]
    #if dss == 1 : keeps = [[0,1], [0,2], [1,2], [0, 3], [1,3]] 
    fig, ax = plt.subplots(1,3, figsize = (3*5,5))
    ax = ax.ravel()
            
    for i, keep in enumerate(keeps):
       
        truth_x, truth_y = parameter_truth[keep[0]], parameter_truth[keep[1]]
        label_x, label_y = parameter_name[keep[0]], parameter_name[keep[1]]

        extent_x = extent[keep[0]]
        extent_y = extent[keep[1]]

        
        Covlist_mar = []
        
        for j in range(len(fisher_params_filename)):
            data = np.loadtxt(fisher_params_filename[j]) 
            row_x, row_y = data.shape
            N_params = int(np.sqrt(row_x))
        
        
            mask1 = np.zeros((N_params,N_params), dtype=bool)
            mask2 = np.zeros((N_params,N_params), dtype=bool)
            mask1[:,keep] = 1
            mask2[keep,:] = 1
            mask = mask1*mask2

            if dss == 1 : 
                mask_dss_p = np.ones((N_params,N_params), dtype=bool)
                mask_dss_p[:,3] = 0
                mask_dss_p[3,:] = 0


                mask_dss_xi = np.ones((N_params,N_params), dtype=bool)
                mask_dss_xi[:,2] = 0
                mask_dss_xi[2,:] = 0

                Fisherlist = [ data[:,q].reshape(N_params,N_params) for q in range(1,row_y)]


                CovP = inv(Fisherlist[0][mask_dss_p].reshape(3,3))
                CovXi = inv(Fisherlist[1][mask_dss_xi].reshape(3,3))
                Covtot_p = inv(Fisherlist[2])[mask_dss_p].reshape(3,3) # keep v_p (v_xi marginalized)
                Covtot_xi = inv(Fisherlist[2])[mask_dss_xi].reshape(3,3)
                Covlist = [CovP, CovXi, Covtot_p, Covtot_xi]
            
            else : Covlist = [ inv(data[:,q].reshape(N_params,N_params)) for q in range(1,row_y)]
            
            if diffsky: 
                C_diffsky = inv((data[:,1]+data[:,2]).reshape(N_params,N_params))
                Covlist += [C_diffsky]
            Covlist_mar += [cov[mask].reshape(2,2) for cov in Covlist]
            
        el = confidence_ellipse(truth_x,truth_y, linestyle, linecolor, *tuple(Covlist_mar) )

        for e in el:
            ax[i].add_artist(e)
            e.set_clip_box(ax[i].bbox)

        xmin = truth_x*extent_x[0]
        xmax = truth_x*extent_x[1]
        ymin = truth_y*extent_y[0]
        ymax = truth_y*extent_y[1]

        ax[i].tick_params(labelsize=15)
        ax[i].locator_params(axis = 'x', nbins=5)
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_ylabel(label_y, size = 20)
        ax[i].set_xlabel(label_x, size = 20)

        labellist = ['P', 'Xi', 'tot']
        ax[i].text(xmin * 1.0005, ymin, text_label, size = 15) 
    
    fig.tight_layout()
    fig.savefig(out_name)    

def plot_reid(filename = None, labels =None, kind = None, color = None, out_name = 'figure/reid_test.png'):

    fig, (ax, ax2) = plt.subplots(2,1, figsize=(7, 7))
    
    
    for i in range(len(filename)):
        if kind[i]: linestyle='-.'
        else : linestyle='-'
            
        data = np.genfromtxt(filename[i])
        rlist, errb, errf = data[:,0], data[:,1], data[:,2]
        
        ax.plot(rlist, errb , linestyle = linestyle, label=labels[i], color = color[i])
        ax2.plot(rlist, errf, linestyle = linestyle, color=ax.lines[-1].get_color() )

    ax.set_xlim(0, 60)
    ax.set_ylim(0.0005, 0.07)
    ax.set_ylabel(r'$\sigma_{b}/b$', size = 20)
    ax.set_xlabel(r'$r_{\rm{min}} (Mpc/h)$', size = 20)
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.tick_params(labelsize=15)


    ax2.set_xlim(0, 60)
    ax2.set_ylim(0.00, 0.08)
    ax2.set_ylabel(r'$\sigma_{f}/f$',size=20)
    ax2.set_xlabel(r'$r_{\rm{min}} (Mpc/h)$', size = 20)
    ax2.tick_params(labelsize=15)

    plt.tight_layout()
    #ax.set_title(' from F_bandpower ')
    #figname = 'figure/reid.pdf'
    #figname = 'figure/reid_n.pdf'
    fig.savefig(out_name)
    print 'fig save to ', out_name
    
def plot_snr(filename = None, labels =None, kind = None, color = None, out_name = 'figure/snr_test.png'):

    
    fig, ax = plt.subplots(1,1, figsize=(7, 5))
    
    
    for i in range(len(filename)):
        if kind[i]: linestyle='-.'
        else : linestyle='-'
            
        data = np.genfromtxt(filename[i])
        klist, snr = data[:,0], data[:,1]
        
        ax.plot(klist, np.sqrt(snr) , linestyle = linestyle, label=labels[i], color = color[i])

    ax.set_xlim(0.05, 2)
    ax.set_ylim(0.0, 8e2)
    ax.set_ylabel(r'$\sqrt{(SNR)^2}$', size = 20)
    ax.set_xlabel(r'$k_{\rm{max}} (h/Mpc)$', size = 20)
    ax.set_xscale('linear')
    ax.legend(loc = 'best')
    ax.tick_params(labelsize=15)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.tight_layout()
    #ax.set_title(' from F_bandpower ')
    #figname = 'figure/reid.pdf'
    #figname = 'figure/reid_n.pdf'
    fig.savefig(out_name)
    print 'fig save to ', out_name
    
def main_example():
    
    
    ## ellipse example
    fisher_params_filename = 'data_txt/default_kN1000_fisher_params.txt'
    plot_single_ellipse(fisher_params_filename, parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5], keep = [0,1]\
                 , text_label = '\n', out_name = 'figure/ellipse_test.png')

    plot_triple_ellipse(fisher_params_filename, parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5]
                 , text_label = '\n', out_name = 'figure/ellipse_test.png')
    
    
    
    ## ellipse plotting more than one data together
    sanchez = 'data_txt/sanchez_fisher_params.txt'
    sanchez_bfsn = 'data_txt/sanchez_bfsn_fisher_params.txt'
    satpathy = 'data_txt/satpathy_fisher_params.txt'
    satpathy_bfsn = 'data_txt/satpathy_bfsn_fisher_params.txt'
    ellipse_test = 'data_txt/ellipse_test_fisher_params.txt'
    ellipse_test_bfsn = 'data_txt/ellipse_test_bfsn_fisher_params.txt'

    plot_triple_ellipse([sanchez, sanchez_bfsn], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5]
                 , text_label = '\n', out_name = 'figure/ellipse_sanchez.png')
    plot_triple_ellipse([satpathy,satpathy_bfsn ], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5]
                 , text_label = '\n', out_name = 'figure/ellipse_satpathy.png')
    plot_triple_ellipse([ellipse_test, ellipse_test_bfsn], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5]
                 , text_label = '\n', out_name = 'figure/ellipse_satpathy.png')

    # ellipse diff sky
    plot_triple_ellipse([sanchez], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                 , parameter_truth = [2.0, 0.74, 3.5], diffsky=1
                 , text_label = '\n', out_name = 'figure/ellipse_sanchez_diffsky.png')
    plot_triple_ellipse([satpathy], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                     , parameter_truth = [2.0, 0.74, 3.5], diffsky=1
                     , text_label = '\n', out_name = 'figure/ellipse_satpathy_diffsky.png')
    plot_triple_ellipse([ellipse_test], parameter_name = [r'$b$', r'$f$', r'$\sigma_v$']\
                     , parameter_truth = [2.0, 0.74, 3.5], diffsky=1
                     , text_label = '\n', out_name = 'figure/ellipse_test_diffsky.png')

    ## Reid example
    filename = ['data_txt/reid/reid_bf_kN1000_rN180_reid_p.txt',
            'data_txt/reid/reid_bfs_kN1000_rN90_reid_p.txt',
            'data_txt/reid/reid_bfsn_kN1000_rN90_reid_p.txt',
            'data_txt/reid/reid_bf_kN1000_rN90_reid_xi.txt',
            'data_txt/reid/reid_bfs_kN1000_rN90_reid_xi.txt',
            'data_txt/reid/reid_bfsn_kN1000_rN90_reid_xi.txt',
            ]
    kind = [1, 1, 1, 0, 0, 0]
    labels = ['', '', '','bf', 'bfs', 'bfsn']
    color = ['blue', 'green', 'red', 'blue', 'green', 'red']
    plot_reid(filename = filename, labels =labels, kind = kind, color=color, out_name =
              'figure/reid_marginalization_comparison.png')


    # Cumulative SNR example
    filename = ['data_txt/snr/reid_bf_kN1000_rN180_snr_p',
            'data_txt/snr/reid_bf_kN1000_rN180_snr_xi',
            'data_txt/snr/reid_bf_kN1000_rN180_snr_tot'
            ]
    kind = [1, 0, 1]
    labels = ['p', 'xi', 'tot']
    color = ['blue', 'green', 'red']
    plot_snr(filename = filename, labels =labels, kind = kind, color=color, 
              out_name = 'figure/snr_test.png')

    
#########################
#main()


