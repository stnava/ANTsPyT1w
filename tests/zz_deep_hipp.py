import sys, os
import unittest
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import tempfile
import shutil
import antspyt1w
import antspynet
import ants
import re
import pandas as pd
import tensorflow as tf
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
img = ants.image_read( fn )
img = img * antspynet.brain_extraction( img, 't1' )
bfn = antspynet.get_antsxnet_data( "croppedMni152" )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
# deep_hippo(img, template, number_of_tries=10, syn_type='antsRegistrationSyNQuickRepro[a]', verbose=False)
mdl = tf.keras.models.load_model( antspyt1w.get_data("SEGSRES_Sharp_32_ANINN222_1_good3", target_extension=".h5" ) )
hippLR = antspyt1w.deep_hippo( img, templateb, 1, tx_type='Affine',
    sr_model=mdl,
    verbose=True )
