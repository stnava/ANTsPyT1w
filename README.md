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
img = ants.image_read( fn )

# generalized default processing
myresults = antspyt1w.hierarchical( img, output_prefix = '/tmp/XXX' )

##### organize summary data into data frames - user should pivot these to columns
# and attach to unique IDs when accumulating for large-scale studies
# see below for how to easily pivot into wide format
# https://stackoverflow.com/questions/28337117/how-to-pivot-a-dataframe-in-pandas


```


## to publish a release

```
python3 -m build
python -m twine upload -u username -p password  dist/*
```
