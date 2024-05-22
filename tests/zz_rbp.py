import ants
import antspyt1w
import antspynet
import pandas as pd
import numpy as np
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
x = ants.image_read( fn )
imgbxt = antspyt1w.brain_extraction( x, method='v1' )
x = antspyt1w.preprocess_intensity( x, imgbxt )
###############################################
bfn = antspynet.get_antsxnet_data( "croppedMni152" )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1v0' ) ).iMath( "Normalize" )
templatesmall = ants.resample_image( templateb, (91,109,91), use_voxels=True )
rbp = antspyt1w.random_basis_projection( x, templatesmall )
