import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_psid_2_factors(latents, dir_index, ground_truth_behaviours, predicted_behaviours):
    
    #Visualize results
    plt.figure(figsize=(16,5))
    colors = plt.cm.nipy_spectral(np.arange(8)/8)

    n_show = latents.shape[0]
    t_show = latents.shape[1]
    t_burn = 0
    dirs_show = set([0,1,2,3,4,5,6,7])

    plt.subplot(1,3,1)
    for t in range(n_show):
        if dir_index[t] in dirs_show:
            plt.plot(ground_truth_behaviours[t,t_burn:t_show,0],ground_truth_behaviours[t,t_burn:t_show,1],color=colors[dir_index[t]],alpha=.75, lw=1)
    plt.title('True behaviour')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1,3,2)
    for t in range(n_show):
        if dir_index[t] in dirs_show:
            plt.plot(predicted_behaviours[t,t_burn:t_show,0],predicted_behaviours[t,t_burn:t_show,1],color=colors[dir_index[t]],alpha=.75,lw=2)

    plt.title('Predicted behaviour')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1,3,3)
    n_show = latents.shape[0]
    for t in range(n_show): 
        if dir_index[t] in dirs_show:
            plt.plot(latents[t,t_burn:t_show,0], latents[t,t_burn:t_show,1], color=colors[dir_index[t]],alpha=1, lw=1)
    plt.title('Behaviourally Relevant factors')
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2');