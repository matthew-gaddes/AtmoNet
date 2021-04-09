#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:41:47 2021

@author: matthew
"""
dependency_paths = {'syinterferopy_bin'  : '/home/matthew/university_work/15_my_software_releases/SyInterferoPy-2.0.1/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'licsar_web_tools'   : '/home/matthew/university_work/15_my_software_releases/LiCSAR-web-tools-1.1.0/',                                      # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'local_scripts'      : '/home/matthew/university_work/python_stuff/python_scripts/'}
                    

import sys
for dependency_name, dependency_path in dependency_paths.items():
    sys.path.append(dependency_path)

#%% Imports


from pathlib import Path
import numpy as np
import glob

from small_plot_functions import matrix_show

from licsar_web_tools_downloading import download_LiCSAR_portal_data
from licsar_web_tools_converting import convert_LiCSAR_portal_data


#%% Things to set


frameID = '014A_07688_131313'                                       # As defined on the LiCSAR portal
date_start = 20180101                                               # YYYYMMDD
date_end   = 20200101                                               # YYYYMMDD
download_metadata = True                                            # the network of baselines, DEM etc.  
between_epoch_files = ['geo.unw.tif', 'geo.unw.png']                 # Possible files: 'geo.cc.png', 'geo.cc.tif', 'geo.diff.png', 'geo.diff_pha.tif', 'geo_diff_unfiltered.png', 'geo_diff_unfiltered_pha.tif', 'geo.unw.png', 'geo.unw.tif'
epoch_files = ['ztd.geo.tif', 'ztd.jpg']                             # Possible files: 'geo.mli.png', 'geo.mli.tif', 'sltd.geo.tif', 'ztd.geo.tif', 'ztd.jpg'
n_para = 4                                                          # Parallelisation.  The number of cores is a good starting point.    
convert_metadata = True

#%% Download the data and convert from geotiff to numpy array

#download_LiCSAR_portal_data(frameID, date_start, date_end, download_metadata, epoch_files, between_epoch_files, n_para)




metadata_nps, epoch_files_nps, between_epoch_files_nps = convert_LiCSAR_portal_data(frameID, convert_metadata, epoch_files, between_epoch_files, mask_vals = [0])


matrix_show(metadata_nps['hgt'])
#%%

# n_samples

def licsar_portal_nps_to_tensors(between_epoch_files_nps, epoch_files_nps, crop_coords = None):
    """ Given dictionaries of LiCSAR portal products, convert these to tensors
    
    Returns:
        X_unw    | n_samples x ny x nx x 1
        X_gacos  | n_samples x ny x nx x 1 
    
    Notes:
        list of metadata to do with each of these?
    """
    
    between_epoch_files = 'geo.unw.tif'
    gacos_file = 'ztd.geo.tif'
    
    
    # 0: get the dates that the interferograms span
    ifg_dates = list(between_epoch_files_nps[between_epoch_files].keys())
    
    
    
    # 1: create X_unw (make a single tensor with all ifgs in it)
    unw1 = between_epoch_files_nps[between_epoch_files][ifg_dates[0]]
    if crop_coords is not None:
        unw1 = unw1[crop_coords['y_start'] : crop_coords['y_stop'], crop_coords['x_start'] : crop_coords['x_stop']]
    (ny, nx) = unw1.shape
    n_ifg = len(ifg_dates)
    X_unw = np.zeros((n_ifg, ny, nx, 1))                                                                                            # initiate
    X_gacos = np.zeros((n_ifg, ny, nx, 1))                                                                                          # initiate
    bad_dates = []                                                                                                                  # intiate a list to store bad dates on (one or more of the gacos files are not available)
    for ifg_n, ifg_date in enumerate(ifg_dates):
        date_1 = ifg_date[:8]
        date_2 = ifg_date[9:]
        
        try:
            gacos_1 = epoch_files_nps[gacos_file][date_1][crop_coords['y_start'] : crop_coords['y_stop'], crop_coords['x_start'] : crop_coords['x_stop']]
            gacos_2 = epoch_files_nps[gacos_file][date_2][crop_coords['y_start'] : crop_coords['y_stop'], crop_coords['x_start'] : crop_coords['x_stop']]
            gacos_available = True
        except:
            gacos_available = False
            #print(f"One or more of the Gacos files are not available for interferogram {ifg_date}.  Continuing anyway.  ")
            
        if gacos_available:                                                                                                                                                     # if Gacos is available, unw and gacos will be written to the tensor
            gacos = gacos_1 - gacos_2                                                                                                                                           # difference gacos for the two ifg dates.  
            unw = between_epoch_files_nps[between_epoch_files][ifg_date][crop_coords['y_start'] : crop_coords['y_stop'], crop_coords['x_start'] : crop_coords['x_stop']]        # get the unw from the dict and crop it      
            X_gacos[ifg_n,:,:,0] = gacos                                                                                                                                        # write to the tensor
            X_unw[ifg_n,:,:,0] = unw                                                                                                                                            # write to the tensor.  
        else:
            bad_dates.append(ifg_date)                                                                                                                                          # if not available, update a list of bad dates.  
            
    # 3: Remove any blanks (that are caused by missing gacos products.  )
    print(f"{len(bad_dates)} interferograms with missing gacos products have been found, and these will be removed.  ")
    for bad_date in bad_dates:
        ifg_n = ifg_dates.index(bad_date)
        ifg_dates.remove(bad_date)                                                                                              # simple remove from list
        X_unw = np.delete(X_unw, ifg_n, 0)                                                                                      # ifgs are first axis, so delete that "row" from the first axis
        X_gacos = np.delete(X_gacos, ifg_n, 0)
        
    return X_unw, X_gacos, ifg_dates
        
crop_coords = {'x_start' : 500,
               'x_stop'  : 2000,
               'y_start' : 500,
               'y_stop'  : 2000}
X_unw, X_gacos, ifg_dates = licsar_portal_nps_to_tensors(between_epoch_files_nps, epoch_files_nps, crop_coords)

#%%
    
def view_atmonet_data(X_unw, X_gacos, ifg_dates, Y = None, Y_pred = None, plot_args = None):
    """Given some unwrapped phase, the gacos correction, and possibly the synthetic deformation and the model recovered deformation, plot all four.  
    """
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
    
view_atmonet_data(X_unw, X_gacos, ifg_dates)
    
    
    
    
#%%
def visualise_UnwrapNet(X, Y, model, n_data = 10):
    """Given some data (X), the labels (Y), and a model, predict the labels on n_data of these, 
    then plot the predictions (and the residuals)
    Inputs:
        X | rank 4 array | n_ims first, channels last.  Usually wrapped data.  
        Y | rank 4 array | n_ims first, channels last.  Usually unwrapped data
        model | keras model | used to predict Y_fcn
        n_data |  int | up to this many data will be plotted
    Returns:
        Matplotlib figure
    History:
        2021_03_03 | MEG | Written.  
    """
    
    import matplotlib.pyplot as plt
    
    if n_data > X.shape[0]:
        n_data = X.shape[0]

    Y_fcn = model.predict(X[:n_data,], verbose = 1)                                # forward pass of the testing data bottleneck features through the fully connected part of the model
    
    fig, axes = plt.subplots(4, n_data)  
    if n_data == 1:    
        axes = np.atleast_2d(axes).T                                                # make 2d, and a column (not a row)
    
    row_labels = ['wrapped (X)', 'Unw. (Y)', 'Unw. model(Y_fcn)', 'Resid.' ]
    for ax, label in zip(axes[:,0], row_labels):
        ax.set_ylabel(label)
    
    
    imshow_settings = {'interpolation' : 'none', 
                       'aspect'        : 'equal'}
    
    for data_n in range(n_data):
        
        # 0 calcuated the min and max values for the unwrapped data.  
        Y_combined = np.concatenate((Y[data_n,], Y_fcn[data_n]), axis = 0)   
        Y_min = np.min(Y_combined)
        Y_max = np.max(Y_combined)
        
        # 1: Do each of the 4 plots
        w = axes[0,data_n].imshow(X[data_n,:,:,0], **imshow_settings)                                                   # wrapped data
        axin = axes[0,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(w, cax=axin, orientation='horizontal')
        
        unw = axes[1,data_n].imshow(Y[data_n,:,:,0], **imshow_settings, vmin = Y_min, vmax = Y_max)                     # unwrapped 
        axin = axes[1,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(unw, cax=axin, orientation='horizontal')
        
        unw_cnn = axes[2,data_n].imshow(Y_fcn[data_n,:,:,0], **imshow_settings, vmin = Y_min, vmax = Y_max)             # unwrapped predicted by the model
        axin = axes[2,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(unw, cax=axin, orientation='horizontal')
        
        resid = axes[3,data_n].imshow((Y[data_n,:,:,0] - Y_fcn[data_n,:,:,0]), **imshow_settings)                       # difference betwee two unwrappeds.  
        axin = axes[3,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(resid, cax=axin, orientation='horizontal')
    
    for ax in np.ravel(axes):
        ax.set_yticks([])
        ax.set_xticks([])
    
#%%
def add_synthetic_deformation_to_tensors():
    """ Given tensors that contain LiCSAR products for use with a Keras model, add synthetic deformaiton patterns to these.  
    
    Returns:
        X_unw
        X_gacos
        Y               # tensor containing the deformation.  
    """
    
    
    


# date1 = list(epoch_files_nps['geo.mli.tif'].keys())[0]                                                             # get the date of one of the images
# f, ax = plt.subplots(1,1)
# ax.imshow(epoch_files_nps['geo.mli.tif'][date1])                                                                    # and plot that date


# date1 = list(between_epoch_files_nps['geo.unw.tif'].keys())[0]                                                     # get the date of one of the images
# f, ax = plt.subplots(1,1)
# ax.imshow(between_epoch_files_nps['geo.unw.tif'][date1])                                                            # and plot that date




#%%

