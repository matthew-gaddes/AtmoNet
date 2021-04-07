#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:41:47 2021

@author: matthew
"""
dependency_paths = {'syinterferopy_bin'  : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/',                                      # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'local_scripts'      : '/home/matthew/university_work/python_stuff/python_scripts/',
                    'LiCSBAS_1.3.9'      : '/home/matthew/university_work/03_automatic_detection_algorithm/07_LiCSBAS/LiCSBAS-1.3.9/bin/'}       #

import sys
for dependency_name, dependency_path in dependency_paths.items():
    sys.path.append(dependency_path)

#%% Imports

import rasterio
from pathlib import Path
import numpy as np
import glob

from small_plot_functions import matrix_show
# #%% Things to set
# frame = '074A_05055_131313'



# #%%

# ifgs = glob.glob(f"./{frame}/interferograms/*/*")

# for ifg in ifgs:
#     bands = rasterio.open(ifg)
#     unw_ph = bands.read()
    
    
    
    
# band_ds = rasterio.open(band_file)                                      # open the Geotiff
# if band_n == 0:                                                         # if it's the first one... 
#     ny, nx = band_ds.read(1).shape                                      # use it to get the size of the data
#     image_data = np.zeros((ny, nx, n_bands))                            # and initiate an array to store the data in,  channels last format.  Note - this won't work if channels with different resolution are used.  
# image_data[:,:, band_n] = band_ds.read(1)                               # read the geotif and add to the array that stores all the data 


#%% Gacos

# import os

# wd = os.getcwd()
# LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'

# gacosdir = os.path.join(wd, 'GACOS')
# if not os.path.exists(gacosdir): os.mkdir(gacosdir)

# ### Get available dates
# print('\nDownload GACOS data', flush=True)
# url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
# response = requests.get(url)
# response.encoding = response.apparent_encoding #avoid garble
# html_doc = response.text
# soup = BeautifulSoup(html_doc, "html.parser")
# tags = soup.find_all(href=re.compile(r"\d{8}"))
# imdates_all = [tag.get("href")[0:8] for tag in tags]
# _imdates = np.int32(np.array(imdates_all))
# _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()
# print('  There are {} epochs from {} to {}'.format(len(_imdates),
#                                startdate, enddate), flush=True)

# ### Extract available dates
# print('  Searching available epochs ({} parallel)...'.format(n_para), flush=True)

# args = [(i, len(_imdates),
#          os.path.join(url, imd, '{}.sltd.geo.tif'.format(imd)),
#          os.path.join(gacosdir, imd+'.sltd.geo.tif')
#          ) for i, imd in enumerate(_imdates)]

# p = multi.Pool(n_para)
# rc = p.map(check_gacos_wrapper, args)
# p.close()

# n_im_existing = 0
# n_im_unavailable = 0
# imdates_dl = []
# for i, rc1 in enumerate(rc):
#     if rc1 == 0:  ## No need to download
#         n_im_existing = n_im_existing + 1
#     if rc1 == 3 or rc1 == 5:  ## Can not download
#         n_im_unavailable = n_im_unavailable + 1
#     elif rc1 == 1 or rc1 == 2  or rc1 == 4:  ## Need download
#         imdates_dl.append(_imdates[i])

# n_im_dl = len(imdates_dl)

# if n_im_existing > 0:
#     print('  {} GACOS data already downloaded'.format(n_im_existing), flush=True)
# if n_im_unavailable > 0:
#     print('  {} GACOS data unavailable'.format(n_im_unavailable), flush=True)

# ### Download
# if n_im_dl > 0:
#     print('{} GACOS data will be downloaded'.format(n_im_dl), flush=True)
#     print('Download GACOS ({} parallel)...'.format(n_para), flush=True)
#     ### Download
#     args = [(i, imd, n_im_dl,
#              os.path.join(url, imd, '{}.sltd.geo.tif'.format(imd)),
#              os.path.join(gacosdir, '{}.sltd.geo.tif'.format(imd))
#              ) for i, imd in enumerate(imdates_dl)]
    
#     p = multi.Pool(n_para)
#     p.map(download_wrapper, args)
#     p.close()
# else:
#     print('No GACOS data available from {} to {}'.format(startdate, enddate), flush=True)


#%%

# import subprocess
# frameID = '074A_05055_131313'
# date_start = 20190901
# date_end = 20200101

# # #LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos] [--n_para int]

# subprocess.call(f"LiCSBAS01_get_geotiff.py -f {frameID} -s {date_start} -e {date_end} --get_gacos {True}" , shell=True)                 # 

# # #subprocess.call(f"LiCSBAS02_ml_prep.py -i {GEOCdir} -o {GEOCmldir} -n {downsampling} --n_para {n_para}" + f" >&1 | tee -a {logfile_dir}LiCSBAS_log.txt", shell=True)                 # This creates the files in GEOCmlXXX, including the png preview of unw, note that 1 is stdout, -a to append

# import sys; sys.exit()

#%%


#ver=1.6; date=20200911; author="Y. Morishita"
# print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
# print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

#%% Dependencies

from pathlib import Path

import os
import datetime as dt
import requests
from bs4 import BeautifulSoup                                                                            # to get data out of HTML etc.  
import re
import multiprocessing as multi

def comp_size_time(file_remote, file_local):
    """
    Compare size and time of remote and local files.
    Returns:
        0 : Size is identical and local time is new
        1 : Size is not identical
        2 : Size is identical but remote time is new
        3 : Remote not exist
    """
    import requests
    import datetime as dt
    import dateutil

    response = requests.head(file_remote, allow_redirects=True)
    
    if response.status_code != 200:
        return 3

    size_remote = int(response.headers.get("Content-Length"))
    size_local = os.path.getsize(file_local)
    if size_remote != size_local: ### Different size
        return 1

    time_remote = dateutil.parser.parse(response.headers.get("Last-Modified"))
    time_local = dt.datetime.fromtimestamp(os.path.getmtime(file_local), dt.timezone.utc)
    if time_remote > time_local: ### New file
        return 2

    return 0

def download_data(url, file):
    """ Download a file from a url and set the name of the downloaded file.  
    Inputs:
        url | Path | url of file to download (i.e. including the file name)
        file | string | name of file to save as
    Returns:
        file
    History:
        2021_??_?? | YM | Part of LiCSBAS
        2021_03_25 | MEG | Write the docs.   
        
    """
    import time
    import requests
    
    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(np.floor(np.log(size_bytes)/np.log(1024)))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        return "%s%s" % (s, size_name[i])

    try:
       start = time.time() 
       with requests.get(url) as res:
            res.raise_for_status()
            with open(file, "wb") as output:
                output.write(res.content)
       elapsed = int(time.time()-start)
       if elapsed==0: elapsed=elapsed+1
       fsize = convert_size(os.path.getsize(file))
       speed = convert_size(int(os.path.getsize(file)/elapsed))
       print('  {}, {}, {}s, {}/s'.format(os.path.basename(file),
                                          fsize, elapsed, speed))

    except:
        print(
            "    Error while downloading from {}".format(url),
            file=sys.stderr,
            flush=True,
        )
        return


def download_wrapper(args):
    i, ifgd, n_dl, url_data, path_data = args
    dir_data = os.path.dirname(path_data)
    print('  Donwnloading {} ({}/{})...'.format(ifgd, i+1, n_dl), flush=True)
    if not os.path.exists(dir_data): os.mkdir(dir_data)
    download_data(url_data, path_data)

def check_gacos_wrapper(args):
    """
    Returns :
        0 : Local exist, complete, and new (no need to donwload)
        1 : Local incomplete (need to re-donwload)
        2 : Local old (no need to re-donwload)
        3 : Remote not exist  (can not compare, no download)
        4 : Local not exist and remote exist (need to download)
        5 : Local not exist but remote not exist (can not download)
    """
    i, n_data, url_data, path_data = args
    bname_data = os.path.basename(path_data)
    
    if np.mod(i, 10) == 0:
        print("  {0:3}/{1:3}".format(i, n_data), flush=True)

    if os.path.exists(path_data):
        rc = comp_size_time(url_data, path_data)
        if rc == 1:
            print("Size of {} is not identical.".format(bname_data), flush=True)
        elif rc == 2:
            print("Newer {} available.".format(bname_data), flush=True)
        return rc
    else:
        response = requests.head(url_data, allow_redirects=True)
        if response.status_code == 200:
            return 4
        else:
            return 5
   
    

def ifgdates2imdates(ifgdates):
    primarylist = []
    secondarylist = []
    for ifgd in ifgdates:
        primarylist.append(ifgd[:8])
        secondarylist.append(ifgd[-8:])

    imdates = list(set(primarylist+secondarylist)) # set is a unique operator
    imdates.sort()

    return imdates

def check_exist_wrapper(args):
    """
    Returns :
        0 : Local exist, complete, and new (no need to donwload)
        1 : Local incomplete (need to re-donwload)
        2 : Local old (no need to re-donwload)
        3 : Remote not exist  (can not compare, no download)
        4 : Local not exist (need to download)
    """

    i, n_data, url_data, path_data = args
    bname_data = os.path.basename(path_data)
    
#    if np.mod(i, 10) == 0:
#        print("  {0:3}/{1:3}".format(i, n_data), flush=True)

    if os.path.exists(path_data):
        rc = comp_size_time(url_data, path_data)
        if rc == 1:
            print("Size of {} is not identical.".format(bname_data), flush=True)
        elif rc == 2:
            print("Newer {} available.".format(bname_data), flush=True)
        return rc
    else:
        return 4
    


def download_metadata_files(trackID, frameId, outdir, LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'):
    """The LiCSAR portal contains a folder of metadata, including products such as the DEM, or a figure showing the baselines.  
    This function downloads them.  
    Inputs:
        trackID | int | the track ID.  Usually the first part of the frameID
        frameID | str | the frame ID, as used on the COMET LiCS portal.  
        outdir | str of path | folder to save the outputs to
        LiCSARweb | string | parent directory for the COMET LiCS data.  Careful with how handled as it looks like a path, but it's really a URL.  
    Returns:
        E, N, U files.  
        hgt file                DEM
        baselines
        metadata
        network.png
    History:
        2021_03_26 | MEG | Modified from a LiCSBAS script.  
        
    
    """
    

    for ENU in ['E', 'N', 'U', 'hgt']:
        enutif = f"{frameID}.geo.{ENU}.tif"                                                                 # build a string of the name of the file
        url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', enutif)                                 # build the url to where that file would be stored by LiCSARweb
        enutif = outdir / enutif                                                                            # conert from filename to path to file.  
        if os.path.exists(enutif):                                                                          # see if the file exists locally
            rc = comp_size_time(url, enutif)                                                                # if it does, compare the remote (first) and local (second) file.  
            if rc == 0:
                print('{} already exist. Skip download.'.format(enutif), flush=True)
                continue                                                                                    # jumps to next part of for loop so skips download
            elif rc == 3:
                print('{} not available. Skip download.'.format(enutif), flush=True)
                continue                                                                                    # jumps to next part of for loop so skips download
            else:
                if rc == 1:
                    print("Size of {} is not identical.".format(enutif))                                    # download will occur
                elif rc == 2:
                    print("Newer {} available.".format(enutif))                                             # download will occur
        
        print('Download {}'.format(enutif), flush=True)                                                     # update user
        download_data(url, enutif)                                                                          # and download.  enturif is path for download, already converted to a posipath
    
    
    
    print('Download baselines', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')                                # construct the URL for the baselines file
    download_data(url, outdir / 'baselines')                                                                # download the baselines file.  Of the form YYYYMMDD YYYYMMDD perp_baseline temporal_baseline
    
    print('Download metadata.txt', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'metadata.txt')                             # construct the URL for the metadata file
    download_data(url, outdir /  'metadata.txt')                                                            # master date and frame centre acquisition time.  
    
    print('Download network.png', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'network.png')                             # construct the URL for the metadata file
    download_data(url, outdir / 'network.png')                                                             # 





def get_folders_from_LiCSAR_portal(url, double_date = True):
    """On the LiCSAR S1 hub, there are folders of links to each product (e.g. for each epoch the radar intensity, or for interferograms between dates)
    This function returns a list of all these links/dates, and is designed to be used to creat paths to then download these.  
    Inputs:
        url | string | url of webpage with links on.  
        double_date | boolean | True if date/link is in interferogram form (YYYYMMDD_YYYYMMDD), False if in date form (YYYYMMDD)
    Returns:
        imdates_all | list of strings | dates/links of interst.  
    History:
        2021_03_26 | MEG | Modified from a LiCSBAS script.  
    """
    response = requests.get(url)                                                                                # get the webpage with all the epochs on
    response.encoding = response.apparent_encoding                                                              # avoid garble?
    html_doc = response.text                                                                                    # turns the text into a giant string.  
    soup = BeautifulSoup(html_doc, "html.parser")                                                               # break the string into something that looks more like normal html
    tags = soup.find_all(href=re.compile(r"\d{8}"))                                                             # get just the links (which are epochs) from the html
    if double_date:
        character_end = 17                                                                                      # double date is of form YYYYMMDD_YYYYMMDD to need 17 characters.  
    else:
        character_end = 8                                                                                       # single date is of form YYYYMMDD so need 8 characters
    imdates_all = [tag.get("href")[0:character_end] for tag in tags]                                            # get the epochs (as a list)
    # _imdates = np.int32(np.array(imdates_all))                                                                # convert to numpy array
    # _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()                   # back to list?
    return imdates_all




def download_LiCSAR_files(remote_dir, local_dir, date_start, date_end, file_extension, double_date = True):
    """ A function to download  any type of file that is either processed per epoch (e.g. Gacos files), or between epochs (e.g. unwrapped phase.  )
    Modified from scripts contained in LiCSBAS.  
    
    Inputs:
        remote_dir | string | the remote directory with the dates for each folder in.  e.g. http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/14/014A_04939_131313/interferograms'
                              Note that this is handled as a string as it's not dependent on the users' OS (i.e. it's aways coming from a linux server)
        local_dir | string (or Path) |  the local directory with the dates for each folder in'014A_04939_131313/interferograms'.  Caution if using strings for paths!
        date_start | string | YYYYMMDD form
        date_end | string | YYYYMMDD form
        file_extension | string | e.g. 'geo.unw.tif'  No . at the start.  
        double_date | boolean | set to True if working with interfeograms, but false if Epochs.  
    Returns:
        Local files.  
    History:
        2021_03_26 | MEG | Created from modified LiCSBAS scripts.  
        
    """
    
    date_start = int(date_start)                                                                        # convert from a string to an int
    date_end = int(date_end)
    
    # 0: Get the dates required:
    file_dates_all = get_folders_from_LiCSAR_portal(remote_dir, double_date = double_date)              # get all the dates available on the portal
    file_dates = []                                                                                     # initate for the dates we want
    for file_date in file_dates_all:
        if double_date:
            mimd = int(file_date[:8])
            simd = int(file_date[-8:])
            if mimd >= date_start and simd <= date_end:
                file_dates.append(file_date)
        else:
            imd = int(file_date[:8])
            if imd >= date_start and imd <= date_end:
                file_dates.append(file_date)
                
    # 1:  Determine which files need to be downloaded
    
    n_files= len(file_dates)
    args = [(i, n_files,
             os.path.join(remote_dir, file_date, f"{file_date}.{file_extension}"),
             os.path.join(local_dir, file_date, f"{file_date}.{file_extension}")
             ) for i, file_date in enumerate(file_dates)]

    p = multi.Pool(n_para)
    rc = p.map(check_exist_wrapper, args)                                                               # compare local (if it exists) and remote.  
    p.close()

    

    n_files_existing = 0
    files_dl = []
    for i, rc1 in enumerate(rc):                                                                    # rc is a list of the status for each file.  Loop through it.  
        if rc1 == 0:                                                                                  ## No need to download
            n_files_existing += 1
        if rc1 == 3 or rc1 == 5:                                                                     ## Can not download
            print(f' {file_dates[i]}.{file_extension} not available.', flush=True)
        elif rc1 == 1 or rc1 == 2  or rc1 == 4:                                                      ## Need download
            files_dl.append(file_dates[i])                                                              # if it does need downloading, append to list

    #import pdb; pdb.set_trace()   

    n_files_dl = len(files_dl)                                                                          # total number of files that will need to be downloaded
    print(f'{n_files_existing} {file_extension} files already downloaded', flush=True)
    print(f'{n_files_dl} {file_extension} files will be downloaded', flush=True)
    

    # 2: Do the downloading:
    if n_files_dl != 0:
        print('Downloading files ({} parallel)...'.format(n_para), flush=True)
        args = [(i, ifgd, n_files_dl,
                 os.path.join(remote_dir, file_dl, f"{file_dl}.{file_extension}"),
                 os.path.join(local_dir, file_dl, f"{file_dl}.{file_extension}")
                 ) for i, file_dl in enumerate(files_dl)]
        
        p = multi.Pool(n_para)
        p.map(download_wrapper, args)
        p.close()
        
    

#%% Set default
frameID = '074A_05055_131313'                                                                   # Central Spain
frameID = '014A_04939_131313'                                                                   # North Anatolian Fault
startdate = 20141001
enddate = int(dt.date.today().strftime("%Y%m%d"))
get_gacos = True
n_para = 4

# it expects you to run the file in te frames dir.  I ahven't done that so it's all a mess.  

#%% Determine frameID
# wd = os.getcwd()
# #wd = f"./{frameID}/"
# if not os.path.exists(wd): os.mkdir(wd)

trackID = str(int(frameID[0:3]))


#%% Directory and file setting


LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'


directories_dict = {'frame'           : Path(f"./{frameID}")}                                              # this is the same struture as the COMET LiCS portal.  
directories_dict = {'interferograms'  : directories_dict['frame'] / 'interferograms',
                    'epochs'          : directories_dict['frame'] / 'epochs',
                    'metadata'        : directories_dict['frame'] / 'metadata'}
                    
for directory in directories_dict.values():
    if not os.path.exists(directory): os.makedirs(directory)                                            # note use of makedirs to required intermediate directoryies can be made if needed.  




#%% Download the ENU and hgt files

download_metadata_files(trackID, frameID, directories_dict['metadata'])

#%% Download te Gacos file and the intererograms.  

url_ifgdir = os.path.join(LiCSARweb, trackID, frameID, 'interferograms')
download_LiCSAR_files(url_ifgdir, directories_dict['interferograms'], '20190101', '20190301', 'geo.unw.png')

url_epochdir = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
download_LiCSAR_files(url_epochdir, directories_dict['epochs'], '20190101', '20200301', 'ztd.jpg', double_date=False)

import sys; sys.exit()

#%% mli




#%%



#%% unw and cc
### Get available dates


