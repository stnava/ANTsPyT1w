"""
Get local ANTsPyT1w data
"""

__all__ = ['get_data']

from pathlib import Path
import os
import tensorflow as tf

DATA_PATH = os.path.expanduser('~/.antspyt1w/')

def get_data(name=None):
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
            - 'T_template0'
            - 'T_template0_LR'
            - 'T_template0_LobesBstem'
            - 'T_template0_Symmetric'
            - 'T_template0_SymmetricLR'
            - 'PPMI-3803-20120814-MRI_T1-I340756'
            - 'simwmseg'
            - 'simwmdisc'

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

    files = []
    for fname in os.listdir(DATA_PATH):
        if (fname.endswith('.nii.gz')) or (fname.endswith('.jpg') or (fname.endswith('.csv'))):
            fname = os.path.join(DATA_PATH, fname)
            files.append(fname)

    if len( files ) == 0:
        url = "https://ndownloader.figshare.com/articles/14766102/versions/4"
        target_file_name = "14766102.zip"
        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
            cache_subdir=DATA_PATH, extract = True )
        for fname in os.listdir(DATA_PATH):
            if (fname.endswith('.nii.gz')) or (fname.endswith('.jpg') or (fname.endswith('.csv'))):
                fname = os.path.join(DATA_PATH, fname)
                files.append(fname)

    if name == 'all':
        return files

    datapath = None

    for fname in os.listdir(DATA_PATH):
        mystem = (Path(fname).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        mystem = (Path(mystem).resolve().stem)
        if (name == mystem ) :
            datapath = os.path.join(DATA_PATH, fname)

    if datapath is None:
        raise ValueError('File doesnt exist. Options: ' , os.listdir(DATA_PATH))
    return datapath
