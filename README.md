# superiq

## super-resolution image quantitation

make super-resolution fly: quantitatively and at scale

### quantitative methods for super-resolution images

* task-specific super-resolution (SR)
    * global tissue segmentation with priors
    * local cortical segmentation
    * hippocampus segmentation
    * basal forebrain segmentation
    * deep brain structure segmentation
        * substantia nigra
        * caudate, putamen, etc

* general purpose methodology:
    * simultaneous SR for intensity and segmentation probability pairs
    * super-resolution multi-atlas segmentation (SRMAS)
    * local joint label fusion for arbitrary segmentation libraries
    * local joint label fusion for single templates with augmentation

tests provide a good example of use cases.

* install: `python setup.py install`

* test: `python tests/test_segmentation.py`


## Regions

### Eisai regions

* group 1: basal forebrain 75L, 76R

* group2: hippocampus
    * use deep hippocampus but 48L, 47R

* group 3: cortex
    * entorhinal: 117L, 116R
    * parahippocampal:  171L 170R
    * middle temporal gyrus: 155L 154R
    * fusiform gyrus:  123L 122R

###  PD regions for MJFF+PPMI

* total intracranial volume (ICV),
* caudate, => 37L, 36R
* putamen => 58L, 57R
* substantia nigra (SN) => is within 62L and 61R but should be dealt with separately
* globus pallidus => 56L, 55R
* corticospinal tract => TODO via registration from template
* supplementary motor area ( 183=left, 182=right - actually a subset of this but misses medial portion )
* `wlab = [36,37,57,58,61,62,55,56,183,182]`


## TODO

* documentation

* testing
    * figure out how to distribute sr models

* ....
