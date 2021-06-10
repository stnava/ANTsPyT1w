import os.path
from os import path

import sys
import ants
import numpy as np
import random
import functools
from operator import mul
from sklearn.utils.extmath import randomized_svd
import ia_batch_utils as batch
import pandas as pd
from multiprocessing import Pool

# for repeatability
np.random.seed(42)

def myproduct(lst):
    return( functools.reduce(mul, lst) )


def main(input_config):
    random.seed(0)
    c = input_config
    nvox = c.nvox
    nBasis = c.nbasis
    X = np.random.rand( nBasis*2, myproduct( nvox ) )

    U, Sigma, randbasis = randomized_svd(
        X,
        n_components=nBasis,
        random_state=None
    )
    if randbasis.shape[1] != myproduct(nvox):
        raise ValueError("columns in rand basis do not match the nvox product")

    randbasis = np.transpose( randbasis )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    if hasattr(c, 'template_bucket'):
        templatefn = batch.get_s3_object(c.template_bucket, c.template_key, 'data')
    imgfn = batch.get_s3_object(c.input_bucket, c.input_value, 'data')
    img = ants.image_read(imgfn).iMath("Normalize")

    #imgt = ants.threshold_image(img, .5, 1)
    #labs = ants.label_geometry_measures(imgt)
    #labs = labs[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]
    #labs_records = labs.to_dict('records')

    #split = c.input_value.split('/')[-1].split('-')
    #rec = {}
    #rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
    #rec['batchid'] = c.batch_id
    #rec['hashfields'] = ['originalimage', 'process', 'batchid', "data"]
    #rec['project'] = split[0]
    #rec['subject'] = split[1]
    #rec['date'] = split[2]
    #rec['modality'] = split[3]
    #rec['repeat'] = split[4]
    #rec['process'] = 'bxt'
    #rec['name'] = "wholebrain"
    #rec['extension'] = ".nii.gz"
    #rec['resolution'] = "OR"

    #for r in labs_records:
    #    label = r['Label']
    #    r.pop('Label', None)
    #    for k,v in r.items():
    #        data_field = {
    #            "label": label,
    #            'key': k,
    #            "value": v,
    #        }
    #        rec['data'] = data_field
    #        batch.write_to_dynamo(rec)




    norm = ants.iMath(img, 'Normalize')
    resamp = ants.resample_image(norm, nvox, use_voxels=True)
    if hasattr(c, 'registration_transform') and hasattr(c, "template_bucket"):
        accepted_transforms = ["Rigid", "Affine", "Similarity", "SyN"]
        if c.registration_transform in accepted_transforms:
            registration_transform = c.registration_transform
        else:
            raise ValueError(f"Expected registration_transform values [{*accepted_transforms,}], not {c.registration_transform}")
        template = ants.image_read( templatefn ).crop_image().resample_image(nvox, use_voxels=True)
        resamp = ants.registration( template, resamp, registration_transform, aff_metric='GC' )['warpedmovout']
        resamp = ants.rank_intensity( resamp )
    imat = ants.get_neighborhood_in_mask(resamp, resamp*0+1,[0,0,0], boundary_condition='mean' )
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    imgsum = resamp.sum()

    record = {}
    uproj_counter = 0
    for i in uproj[0]:
        uproj_counter += 1
        name = "RandBasisProj" + str(uproj_counter).zfill(2)
        record[name] = i
    uprojpos_counter = 0
    for i in uprojpos[0]:
        uprojpos_counter += 1
        name = "RandBasisProjPos" + str(uprojpos_counter).zfill(2)
        record[name] = i
    df = pd.DataFrame(record, index=[0])

    split = c.input_value.split('/')[-1].split('-')
    rec = {}
    rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
    rec['batchid'] = c.batch_id
    rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
    rec['project'] = split[0]
    rec['subject'] = split[1]
    rec['date'] = split[2]
    rec['modality'] = split[3]
    rec['repeat'] = split[4]
    rec['process'] = 'random_basis_projection'
    rec['version'] = c.version
    rec['name'] = "randbasisproj"
    rec['extension'] = ".nii.gz"
    rec['resolution'] = "OR"


    fields = [i for i in df.columns if i.startswith('RandBasis')]
    records = df[fields]
    records = records.to_dict('records')
    for r in records:
        for k,v in r.items():
            data_field = {
                "label": 0,
                'key': k,
                "value": v,
            }
            rec['data'] = data_field
            batch.write_to_dynamo(rec)


if __name__ == "__main__":
    config = sys.argv[1]
    c = batch.LoadConfig(config)
    main(c)
