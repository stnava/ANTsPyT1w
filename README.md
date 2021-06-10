# ANTsPyT1w

## reference processing for t1-weighted neuroimages (human)

keep track of preferred algorithm variations

```
python setup.py install
```

# what this will do

* provide example data

* brain extraction

* denoising

* n4 bias correction

* tissue segmentation

* brain parcellation into hemispheres, lobes and regions

* hippocampus specific segmentation

* t1 hypointensity segmentation and classification

* hypothalamus segmentation

* deformable registration with recommended parameters (after above processing)


```
import antspyt1w
import antspynet
import ants
fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756')
tfn = antspyt1w.get_data('T_template0')
bfn = antspynet.get_antsxnet_data( "biobank" )

templatea = ants.image_read( tfn )
templatea = templatea * antspynet.brain_extraction( templatea, 't1' )
templateb = ants.image_read( bfn )
templateb = templateb * antspynet.brain_extraction( templateb, 't1' )
img = ants.image_read( fn )
img = img * antspynet.brain_extraction( img, 't1' )
mydkt = antspyt1w.deep_dkt( img, templateb )
```
