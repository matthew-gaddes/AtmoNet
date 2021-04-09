#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:11:46 2021

@author: matthew
"""

#%%


def view_atmonet_data(X_unw, X_gacos, ifg_dates, Y = None, Y_pred = None, plot_args = None):
    """Given some unwrapped phase, the gacos correction, and possibly the synthetic deformation and the model recovered deformation, plot all four.  
    """
    import numpy as np
    
    def plot_image_in_axes(im, ax):
        """ Given an image and an axe, plot it with a colourboar across the bottom.  
        """
        imshow_settings = {'interpolation' : 'none', 
                           'aspect'        : 'equal'}
        data = ax.imshow(im, **imshow_settings)                                                   # unwarpped data in row 1
        axin = ax.inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(data, cax=axin, orientation='horizontal')
    
    import matplotlib.pyplot as plt
    
    if plot_args is None:
        plot_args = np.arange(10)                                                                                               # just plot the first ten
    n_plot = plot_args.shape[0]                                                                                                 # each image will be a column
    
    # 0: Initiate the figure and set labels etc.  
    fig, axes = plt.subplots(4, n_plot, figsize = (16,8))                                                                                       # initiate a figure with the correct number of columns 
    row_labels = ['X_unw', 'X_gacos', 'Y', 'Y_nn.' ]
    for ax, label in zip(axes[:,0], row_labels):
        ax.set_ylabel(label)
    for ax in np.ravel(axes):
        ax.set_yticks([])
        ax.set_xticks([])
    
    
    # 1 main loop that does the image plotting    
    for plot_n, plot_arg in enumerate(plot_args):
        axes[0,plot_n].set_title(f"{ifg_dates[plot_arg][:9]} \n{ifg_dates[plot_arg][9:]}")                                    # primary to seconday (master to slave) dates
        
        plot_image_in_axes(X_unw[plot_arg, :,:,0], axes[0, plot_n])                                                 # first row is unwrapped data
        plot_image_in_axes(X_gacos[plot_arg, :,:,0], axes[1, plot_n])                                               # second row is gacos correction
        if Y is not None:
            plot_image_in_axes(Y[plot_arg, :,:,0], axes[2, plot_n])                                                 # third row is (synthetic) deformation
        else:
            axes[2, plot_n].axis('off')
        if Y_pred is not None:
            plot_image_in_axes(Y_pred[plot_arg, :,:,0], axes[3, plot_n])                                            # fourth row is models attempt at recovering deformation.  
        else:
            axes[3, plot_n].axis('off')
    

    

#%%