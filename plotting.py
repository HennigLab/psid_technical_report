import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_psid_2_factors(latents, dir_index, ground_truth_behaviours, predicted_behaviours, A, titles):
    
    #Visualize results
    plt.figure(figsize=(20,5))
    colors = plt.cm.nipy_spectral(np.arange(8)/8)

    n_show = latents.shape[0]
    t_show = latents.shape[1]
    t_burn = 0
    dirs_show = set([0,1,2,3,4,5,6,7])

    plt.subplot(1,4,1)
    for t in range(n_show):
        if dir_index[t] in dirs_show:
            plt.plot(ground_truth_behaviours[t,t_burn:t_show,0],ground_truth_behaviours[t,t_burn:t_show,1],color=colors[dir_index[t]],alpha=.75, lw=1)
    plt.title(titles[0])
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1,4,2)
    for t in range(n_show):
        if dir_index[t] in dirs_show:
            plt.plot(predicted_behaviours[t,t_burn:t_show,0],predicted_behaviours[t,t_burn:t_show,1],color=colors[dir_index[t]],alpha=.75,lw=2)

    plt.title(titles[1])
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1,4,3)
    n_show = latents.shape[0]
    for t in range(n_show): 
        if dir_index[t] in dirs_show:
            plt.plot(latents[t,t_burn:t_show,0], latents[t,t_burn:t_show,1], color=colors[dir_index[t]],alpha=1, lw=1)
    plt.title(titles[2])
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2');
    
    # (Example 1) Eigenvalues when only learning behaviorally relevant states
    plt.subplot(1,4,4)
    idEigs1 = np.linalg.eig(A)[0]
    ax = plt.gca()
    ax.add_patch(patches.Circle((0,0), radius=1, fill=False, color='black', alpha=0.2, ls='-') )
    plt.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')
    plt.scatter(np.real(idEigs1), np.imag(idEigs1), marker='x', facecolors='#00aa00', label='PSID Identified (stage 1)')
    plt.title(titles[3])
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)