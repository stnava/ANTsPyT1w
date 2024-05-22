import ants
import antspyt1w
import antspynet
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
x = ants.image_read( fn )
tfn = antspyt1w.get_data('T_template0', target_extension='.nii.gz' )
tfnw = antspyt1w.get_data('T_template0_WMP', target_extension='.nii.gz' )
bfn = antspynet.get_antsxnet_data( "croppedMni152" )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1v0' ) ).iMath( "Normalize" )
templatea = ants.image_read( tfn )
templatea = ( templatea * antspynet.brain_extraction( templatea, 't1v0' ) ).iMath( "Normalize" )
templateawmprior = ants.image_read( tfnw )
imgbxt = antspyt1w.brain_extraction( x )
img = antspyt1w.preprocess_intensity( x, imgbxt )
myparc = antspyt1w.deep_brain_parcellation( img, templateb,
        do_cortical_propagation = False, verbose=False )

myhypo = antspyt1w.t1_hypointensity(
        img,
        myparc['tissue_segmentation'], # segmentation
        myparc['tissue_probabilities'][3], # wm posteriors
        templatea,
        templateawmprior )
# ants.plot( img, myhypo['wmh_probability_image'] , axis=2)
