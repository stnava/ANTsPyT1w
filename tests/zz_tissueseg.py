import ants
import antspyt1w
import antspynet
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
x = ants.image_read( fn )
bfn = antspynet.get_antsxnet_data( "croppedMni152" )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1v0' ) ).iMath( "Normalize" )
prepro=True
if prepro:
    imgbxt = antspyt1w.brain_extraction( x, method='v1' )
    x = antspyt1w.preprocess_intensity( x, imgbxt )
myparc = antspyt1w.deep_tissue_segmentation( x, templateb  )
