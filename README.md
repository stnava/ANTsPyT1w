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

* deformable registration with recommended parameters (after above processing) **FIXME**


```
import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import antspyt1w
import antspynet
import ants

##### get example data + reference templates
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756')
tfn = antspyt1w.get_data('T_template0')
tlrfn = antspyt1w.get_data('T_template0_LR')
bfn = antspynet.get_antsxnet_data( "biobank" )

##### read images and do simple bxt ops
bxtmethod = 't1combined[5]' # better for individual subjects
# bxtmethod = 't1' # good for templates
templatea = ants.image_read( tfn )
templatea = ( templatea * antspynet.brain_extraction( templatea, 't1' ) ).iMath( "Normalize" )
templatealr = ants.image_read( tlrfn )
templateb = ants.image_read( bfn )
templateb = ( templateb * antspynet.brain_extraction( templateb, 't1' ) ).iMath( "Normalize" )
img = ants.image_read( fn )
imgbxt = antspynet.brain_extraction( img, bxtmethod ).threshold_image(2,3).iMath("GetLargestComponent")
img = img * imgbxt
# quick look at result
ants.plot(img,axis=2,ncol=8,nslices=24 ) # ,filename=fignameL) # or write to file

##### intensity modifications
img = ants.iMath( img, "Normalize" )
img = ants.denoise_image( img, imgbxt, noise_model='Rician')
img = ants.n4_bias_field_correction( img )

##### hierarchical labeling
myparc = antspyt1w.deep_brain_parcellation( img, templateb )

##### specialized labeling
hippLR = antspyt1w.deep_hippo( img, templateb )
# FIXME hypothalamus
# FIXME wmh

##### a computationally costly registration as a catch-all complement to above
# FIXME hemi-reg

```
