# ANTsPyT1w

## reference processing for t1-weighted neuroimages (human)

keep track of preferred algorithm variations

```
python setup.py install
```

# what this will do

* provide example data [check]

* brain extraction [check]

* denoising [check]

* n4 bias correction [check]

* brain parcellation into tissues, hemispheres, lobes and regions [check]

* hippocampus specific segmentation [check]

* t1 hypointensity segmentation and classification **FIXME**

* hypothalamus segmentation **FIXME**

* deformable registration with recommended parameters (after above processing)

# example processing

```python
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import antspyt1w
import antspynet
import ants

##### get example data + reference templates
# NOTE:  PPMI-3803-20120814-MRI_T1-I340756 is a good example of our naming style
# Study-SubjectID-Date-Modality-UniqueID
# where Modality could also be measurement or something else
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756')
tfn = antspyt1w.get_data('T_template0')
tfnw = antspyt1w.get_data('T_template0_WMP')
tlrfn = antspyt1w.get_data('T_template0_LR')
bfn = antspynet.get_antsxnet_data( "croppedMni152" )

##### read images and do simple bxt ops
templatea = ants.image_read( tfn )
templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
templatealr = ants.image_read( tlrfn )
templateawmprior = ants.image_read( tfnw )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
img = ants.image_read( fn )
imgbxt = antspyt1w.brain_extraction( img )
img = img * imgbxt

# optional - quick look at result
# ants.plot(img,axis=2,ncol=8,nslices=24, filename="/tmp/temp.png" )

# this is an unbiased method for identifying predictors that can be used to
# rank / sort data into clusters, some of which may be associated
# with outlierness or low-quality data ... should be saved for each
# image in order to assist QC
templatesmall = ants.resample_image( templateb, (91,109,91), use_voxels=True )
rbp = antspyt1w.random_basis_projection( img, templatesmall, 10 )

# assuming data is reasonable quality, we should proceed with the rest ...


##### intensity modifications
img = ants.iMath( img, "Normalize" )
img = ants.denoise_image( img, imgbxt, noise_model='Rician')
img = ants.n4_bias_field_correction( img ).iMath("Normalize")

##### hierarchical labeling - also outputs SNR measurements relevant to QC
mylr = antspyt1w.label_hemispheres( img, templatea, templatealr )
myparc = antspyt1w.deep_brain_parcellation( img, templateb )

##### organize summary data into data frames - user should pivot these to columns
# and attach to unique IDs when accumulating for large-scale studies
hemi = antspyt1w.map_segmentation_to_dataframe( "hemisphere", myparc['hemisphere_labels'] )
tissue = antspyt1w.map_segmentation_to_dataframe( "tissues", myparc['tissue_segmentation'] )
dktl = antspyt1w.map_segmentation_to_dataframe( "lobes", myparc['dkt_lobes'] )
dktp = antspyt1w.map_segmentation_to_dataframe( "dkt", myparc['dkt_parcellation'] )

##### traditional deformable registration as a high-resolution output that
# will allow us to "fill out" any regions we may want in the future and also
# do data-driven studies
reg = antspyt1w.hemi_reg(
    input_image = img,
    input_image_tissue_segmentation = myparc['tissue_segmentation'],
    input_image_hemisphere_segmentation = mylr,
    input_template = templatea,
    input_template_hemisphere_labels = templatealr,
    output_prefix="/tmp/SYN",
    is_test=True) # set to False for a real run

##### Exploratory: nice to have - t1-based white matter hypointensity estimates
myhypo = antspyt1w.t1_hypointensity( img,
  myparc['tissue_probabilities'][3], # wm posteriors
  templatea,
  templateawmprior )


##### specialized labeling
hippLR = antspyt1w.deep_hippo( img, templateb )

```

## big FIXME: high-level regression test on the outputs above ...
