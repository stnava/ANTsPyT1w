import ants
import sys
import antspynet
import numpy as np
import tensorflow as tf

def deep_hippo(
    img,
    template,
    number_of_tries = 10,
):

    avgleft = img * 0
    avgright = img * 0
    for k in range(number_of_tries):
        rig = ants.registration( template, img,
            "antsRegistrationSyNQuickRepro[a]", random_seed=k )
        rigi = rig['warpedmovout']
        hipp = antspynet.hippmapp3r_segmentation( rigi, do_preprocessing=False )
        hippr = ants.apply_transforms(
            img,
            hipp,
            rig['fwdtransforms'],
            whichtoinvert=[True],
            interpolator='genericLabel',
        )
        avgleft = avgleft + ants.threshold_image( hippr, 2, 2 ) / nLoop
        avgright = avgright + ants.threshold_image( hippr, 1, 1 ) / nLoop


    avgright = ants.iMath(avgright,"Normalize")  # output: probability image right
    avgleft = ants.iMath(avgleft,"Normalize")    # output: probability image left
    hippright_bin = ants.threshold_image( avgright, 0.5, 2.0 ).iMath("GetLargestComponent")
    hippleft_bin = ants.threshold_image( avgleft, 0.5, 2.0 ).iMath("GetLargestComponent")

    hippleftORlabels  = ants.label_geometry_measures(hippleft, avgleft)
    hipprightORlabels  = ants.label_geometry_measures(hippright, avgright)

    labels = {
        'HLProb':avgleft,
        'HLBin':avgleft_bin,
        'HLStats': hippleftORlabels,
        'HRProb':avgright,
        'HRBin':avgright_bin,
        'HRStats': hipprightORlabels,
    }
    return labels
