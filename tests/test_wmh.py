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
ants.plot( img, myhypo['wmh_probability_image'] )
derka
import tensorflow as tf
import numpy as np
xsegmentation=myparc['tissue_segmentation']
xWMProbability=myparc['tissue_probabilities'][3]
template=templatea
templateWMPrior=templateawmprior
x=ants.image_clone( img )
print("Begin Hypo")
mybig = [88,128,128]
templatesmall = ants.resample_image( template, mybig, use_voxels=True )
qaff = ants.registration(
        ants.rank_intensity(x),
        ants.rank_intensity(templatesmall), 'SyN',
        syn_sampling=2,
        syn_metric='CC',
        reg_iterations = [25,15,0,0],
        aff_metric='GC', random_seed=1 )
afftx = qaff['fwdtransforms'][1]
templateWMPrior2x = ants.apply_transforms( x, templateWMPrior, qaff['fwdtransforms'] )
cerebrum = ants.threshold_image( xsegmentation, 2, 4 )
realWM = ants.threshold_image( templateWMPrior2x , 0.1, math.inf )
inimg = ants.rank_intensity( x )
parcellateWMdnz = ants.kmeans_segmentation( inimg, 2, realWM, mrf=0.0 )['probabilityimages'][0]
x2template = ants.apply_transforms( templatesmall, inimg, afftx, whichtoinvert=[True] )
parcellateWMdnz2template = ants.apply_transforms( templatesmall,
      cerebrum * parcellateWMdnz, afftx, whichtoinvert=[True] )
# features = rank+dnz-image, lprob, wprob, wprior at mybig resolution
f1 = x2template.numpy()
f2 = parcellateWMdnz2template.numpy()
f3 = ants.apply_transforms( templatesmall, xWMProbability, afftx, whichtoinvert=[True] ).numpy()
f4 = ants.apply_transforms( templatesmall, templateWMPrior, qaff['fwdtransforms'][0] ).numpy()
myfeatures = np.stack( (f1,f2,f3,f4), axis=3 )
newshape = np.concatenate( [ [1],np.asarray( myfeatures.shape )] )
myfeatures = myfeatures.reshape( newshape )
inshape = [None,None,None,4]
wmhunet = antspynet.create_unet_model_3d( inshape,
        number_of_outputs = 1,
        number_of_layers = 4,
        mode = 'sigmoid' )
wmhunet.load_weights( antspyt1w.get_data("simwmhseg", target_extension='.h5') )
pp = wmhunet.predict( myfeatures )
limg = ants.from_numpy( tf.squeeze( pp[0] ).numpy( ) )
limg = ants.copy_image_info( templatesmall, limg )
lesresam = ants.apply_transforms( x, limg, afftx, whichtoinvert=[False] )
leresam = lesresam * ants.threshold_image(lesresam,0.02,lesresam.max())
