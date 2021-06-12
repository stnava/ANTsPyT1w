"""
Get local ANTsPyT1w data
"""

__all__ = ['get_data']

from pathlib import Path
import os
import pandas as pd
import ants
import tensorflow as tf

import os.path
from os import path
import pickle
import sys
import ants
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds

from multiprocessing import Pool

DATA_PATH = os.path.expanduser('~/.antspyt1w/')

def get_data( name=None, force_download=False, version=12, target_extension='.csv' ):
    """
    Get ANTsPyT1w data filename

    The first time this is called, it will download data to ~/.antspyt1w.
    After, it will just read data from disk.  The ~/.antspyt1w may need to
    be periodically deleted in order to ensure data is current.

    Arguments
    ---------
    name : string
        name of data tag to retrieve
        Options:
            - 'all'
            - 'dkt'
            - 'hemisphere'
            - 'lobes'
            - 'tissues'
            - 'T_template0'
            - 'T_template0_LR'
            - 'T_template0_LobesBstem'
            - 'T_template0_WMP'
            - 'T_template0_Symmetric'
            - 'T_template0_SymmetricLR'
            - 'PPMI-3803-20120814-MRI_T1-I340756'
            - 'simwmseg'
            - 'simwmdisc'
            - 'wmh_evidence'
            - 'wm_major_tracts'

    force_download: boolean

    version: version of data to download (integer)

    Returns
    -------
    string
        filepath of selected data

    Example
    -------
    >>> import ants
    >>> ppmi = ants.get_ants_data('ppmi')
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def download_data( version ):
        url = "https://ndownloader.figshare.com/articles/14766102/versions/" + str(version)
        target_file_name = "14766102.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )

    if force_download == True :
        download_data( version = version )


    files = []
    for fname in os.listdir(DATA_PATH):
        if ( fname.endswith(target_extension) ) :
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len( files ) == 0 :
        download_data( version = version )
        for fname in os.listdir(DATA_PATH):
            if ( fname.endswith(target_extension) ) :
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    if name == 'all':
        return files

    datapath = None

    for fname in os.listdir(DATA_PATH):
        mystem = (Path(fname).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        if ( name == mystem and fname.endswith(target_extension) ) :
            datapath = os.path.join(DATA_PATH, fname)

    if datapath is None:
        raise ValueError('File doesnt exist. Options: ' , os.listdir(DATA_PATH))
    return datapath



def map_segmentation_to_dataframe( segmentation_type, segmentation_image ):
    """
    Match the segmentation to its appropriate data frame.  We do not check
    if the segmentation_type and segmentation_image match; this may be indicated
    by the number of missing values on output eg in column VolumeInMillimeters.

    Arguments
    ---------
    segmentation_type : string
        name of segmentation_type data frame to retrieve
        Options:
            - 'dkt'
            - 'lobes'
            - 'tissues'
            - 'hemisphere'
            - 'wm_major_tracts'

    segmentation_image : antsImage with same values (or mostly the same) as are
        expected by segmentation_type

    Returns
    -------
    dataframe

    """
    mydf_fn = get_data( segmentation_type )
    mydf = pd.read_csv( mydf_fn )
    mylgo = ants.label_geometry_measures( segmentation_image )
    return pd.merge( mydf, mylgo, how='left', on=["Label"] )



def myproduct(lst):
    return( functools.reduce(mul, lst) )


def random_basis_projection( x, template, nBasis=10, random_state = 99 ):

    np.random.seed(int(random_state))
    nvox = template.shape
    X = np.random.rand( nBasis+1, myproduct( nvox ) )
    u, s, randbasis = svds(X, k=nBasis)
    if randbasis.shape[1] != myproduct(nvox):
        raise ValueError("columns in rand basis do not match the nvox product")

    randbasis = np.transpose( randbasis )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    norm = ants.iMath(x,"Normalize")
    resamp = ants.registration( template, norm,
        "Similarity", aff_metric='GC', random_seed=1 )['warpedmovout']
    imat = ants.get_neighborhood_in_mask(resamp, resamp*0+1,[0,0,0], boundary_condition='mean' )
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    record = {}
    uproj_counter = 0
    for i in uproj[0]:
        uproj_counter += 1
        name = "RandBasisProj" + str(uproj_counter).zfill(2)
        record[name] = i
    uprojpos_counter = 0
    for i in uprojpos[0]:
        uprojpos_counter += 1
        name = "RandBasisProjPos" + str(uprojpos_counter).zfill(2)
        record[name] = i
    df = pd.DataFrame(record, index=[0])
    return df
