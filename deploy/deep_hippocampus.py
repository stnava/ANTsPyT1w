import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
from superiq import deep_hippo
import sys
import antspynet
import numpy as np
import ia_batch_utils as batch


def deep_hippo_deploy(input_config):
    c = batch.LoadConfig(input_config)
    if c.environment == "prod":
        input_image_path = batch.get_s3_object(
            c.input_bucket,
            c.input_value,
            "data"
        )
        img = ants.image_read(input_image_path)
        template = antspynet.get_antsxnet_data( "biobank" )
        template = ants.image_read( template )
        template = template * antspynet.brain_extraction( template, 't1v0' )

        sr_model_path = batch.get_s3_object(
            c.model_bucket,
            c.model_key,
            "data"
        )

        if not os.path.exists(c.local_output_path):
            os.makedirs(c.local_output_path)

        input_params = {
            "img": img,
            "template": template,
            "sr_model_path": sr_model_path,
            "output_path": c.local_output_path + '/',
        }

        outputs = deep_hippo(**input_params)

        for key, value in outputs.items():
            split = c.input_value.split('/')[-1].split('-')
            rec = {}
            rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
            rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
            rec['batchid'] = c.batch_id
            rec['project'] = split[0]
            rec['subject'] = split[1]
            rec['date'] = split[2]
            rec['modality'] = split[3]
            rec['repeat'] = split[4]
            rec['process'] = 'deep_hippo'
            rec['version'] = c.version
            rec['name'] = key
            rec['extension'] = ".nii.gz"
            if "OR" in key:
                rec['resolution'] = "OR"
            else:
                rec['resolution'] = "SR"

            df = value[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]
            volumes = df.to_dict('records')
            for r in volumes:
                label = r['Label']
                r.pop("Label", None)
                for k, v in r.items():
                    rec['data'] = {}
                    rec['data']['label'] = label
                    rec['data']['key'] = k
                    rec['data']['value'] = v
                    print(rec)
                    batch.write_to_dynamo(rec)




    if c.environment == "prod":
        batch.handle_outputs(
            c.output_bucket,
            c.output_prefix,
            c.input_value,
            c.process_name,
            c.version,
        )

if __name__ == "__main__":
    config = sys.argv[1]
    deep_hippo_deploy(config)
