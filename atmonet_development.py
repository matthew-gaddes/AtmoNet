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


# date1 = list(epoch_files_nps['geo.mli.tif'].keys())[0]                                                             # get the date of one of the images
# f, ax = plt.subplots(1,1)
# ax.imshow(epoch_files_nps['geo.mli.tif'][date1])                                                                    # and plot that date


# date1 = list(between_epoch_files_nps['geo.unw.tif'].keys())[0]                                                     # get the date of one of the images
# f, ax = plt.subplots(1,1)
# ax.imshow(between_epoch_files_nps['geo.unw.tif'][date1])                                                            # and plot that date




#%%

