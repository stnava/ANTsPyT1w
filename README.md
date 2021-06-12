# ANTsPyT1w

## reference processing for t1-weighted neuroimages (human)

the outputs of these processes can be used for data inspection/cleaning/triage
as well for interrogating neuroscientific hypotheses.

this package also keeps track of the latest preferred algorithm variations for
production environments.

install by calling (within the source directory):

```
python setup.py install
```

or install via `pip install antspyt1w`

# what this will do

- provide example data

- brain extraction

- denoising

- n4 bias correction

- brain parcellation into tissues, hemispheres, lobes and regions

- hippocampus specific segmentation

- t1 hypointensity segmentation and classification *exploratory*

- deformable registration with robust and repeatable parameters

- helpers that organize and annotate segmentation variables into data frames

- hypothalamus segmentation *FIXME/TODO*


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
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
tfn = antspyt1w.get_data('T_template0', target_extension='.nii.gz' )
tfnw = antspyt1w.get_data('T_template0_WMP', target_extension='.nii.gz' )
tlrfn = antspyt1w.get_data('T_template0_LR', target_extension='.nii.gz' )
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
# see below for how to easily pivot into wide format
# https://stackoverflow.com/questions/28337117/how-to-pivot-a-dataframe-in-pandas

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

##### how to use the hemi-reg output to generate any roi value from a template roi
wm_tracts = ants.image_read( antspyt1w.get_data( "wm_major_tracts", target_extension='.nii.gz' ) )
wm_tractsL = ants.apply_transforms( img, wm_tracts, reg['synL']['invtransforms'],
  interpolator='genericLabel' ) * ants.threshold_image( mylr, 1, 1  )
wm_tractsR = ants.apply_transforms( img, wm_tracts, reg['synR']['invtransforms'],
  interpolator='genericLabel' ) * ants.threshold_image( mylr, 2, 2  )
wmtdfL = antspyt1w.map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsL )
wmtdfR = antspyt1w.map_segmentation_to_dataframe( "wm_major_tracts", wm_tractsR )

##### specialized labeling
hippLR = antspyt1w.deep_hippo( img, templateb )

##### Exploratory: nice to have - t1-based white matter hypointensity estimates
myhypo = antspyt1w.t1_hypointensity( img,
  myparc['tissue_probabilities'][3], # wm posteriors
  templatea,
  templateawmprior )

```


## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
