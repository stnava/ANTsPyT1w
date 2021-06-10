import ants
import sys
import antspynet
import numpy as np
import tensorflow as tf

def deep_hippo(
    img,
    template,
    sr_model_path,
    output_path="",
):

    avgleft = img * 0
    avgright = img * 0
    nLoop = 10
    for k in range(nLoop):
        rig = ants.registration( template, img, "Rigid", random_seed=k )
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

    # OR Prob Images
    leftORprob_path = f"{output_path}hippProbORleft.nii.gz"
    rightORprob_path = f"{output_path}hippProbORright.nii.gz"
    ants.image_write(avgleft, leftORprob_path)
    ants.image_write(avgright, rightORprob_path)

    hippright = ants.threshold_image( avgright, 0.5, 10 ).iMath("GetLargestComponent")
    hippleft = ants.threshold_image( avgleft, 0.5, 10.0 ).iMath("GetLargestComponent")

    # OR Seg Images
    leftORseg_path = f"{output_path}hippSegORleft.nii.gz"
    rightORseg_path = f"{output_path}hippSegORright.nii.gz"
    ants.image_write(hippleft, leftORseg_path) # output: segmentation image
    ants.image_write(hippright, rightORseg_path) # output: segmentation image

    hippleftORlabels  = ants.label_geometry_measures(hippleft, avgleft)
    hipprightORlabels  = ants.label_geometry_measures(hippright, avgright)
    hippleftORlabels.to_csv(f'{output_path}hippleftOR-lgm.csv', index=False)
    hipprightORlabels.to_csv(f'{output_path}hipprightOR-lgm.csv', index=False)

    # SR Part
    imglist = []
    seglist = []
    for side in [1,2]:
        if side == 1:
            hippimg = avgright
            sideLabel = "right"
        elif side == 2:
            hippimg = avgleft
            sideLabel = "left"

        mdl = tf.keras.models.load_model(sr_model_path)

        img = ants.iMath( img, "Normalize" ) * 255 - 127.5 # for SR
        hipp1 = ants.threshold_image( hippr, side, side )
        hipp1m = ants.iMath(hipp1,'MD',10)
        imgc = ants.crop_image(img,hipp1m)
        imgch = ants.crop_image(hippimg,hipp1m)
        myarr = np.stack( [imgc.numpy(),imgch.numpy()* 255 - 127.5],axis=3 )
        newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
        myarr = myarr.reshape( newshape )
        pred = mdl.predict( myarr )
        imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
        imgsr = ants.copy_image_info( imgc, imgsr )
        newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
        ants.set_spacing( imgsr,  newspc )
        imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
        imgsrh = ants.copy_image_info( imgc, imgsrh )
        ants.set_spacing( imgsrh,  newspc )
        imglist.append( imgsr )
        seglist.append( imgsrh )
        imgsrhb = ants.threshold_image( imgsrh, 0.5, 1.0 ).iMath("GetLargestComponent")


    rightSR = ants.iMath(imglist[0], "Normalize")  # output: sr image right
    leftSR = ants.iMath(imglist[1], "Normalize")  # output: sr image left
    ants.image_write(rightSR, f"{output_path}hippImgSRright.nii.gz")
    ants.image_write(leftSR, f"{output_path}hippImgSRleft.nii.gz")


    avgrightSR = seglist[0]  # output: probability image right
    avgleftSR = seglist[1]     # output: probability image left
    ants.image_write(avgrightSR, f"{output_path}hippProbSRright.nii.gz")
    ants.image_write(avgleftSR, f"{output_path}hippProbSRleft.nii.gz")


    hipprSRright = ants.threshold_image( avgrightSR, 0.5, 10 ).iMath("GetLargestComponent")
    hipprSRleft = ants.threshold_image( avgleftSR, 0.5, 10.0 ).iMath("GetLargestComponent")
    ants.image_write(
        hipprSRright,
        f"{output_path}hippSegSRright.nii.gz"
    )
    ants.image_write(
        hipprSRleft,
        f"{output_path}hippSegSRleft.nii.gz"
    )

    hippleftSRlabels  = ants.label_geometry_measures(hipprSRleft, avgleftSR)
    hipprightSRlabels  = ants.label_geometry_measures(hipprSRright, avgrightSR)
    hippleftSRlabels.to_csv(f'{output_path}hippleftSR-lgm.csv')
    hipprightSRlabels.to_csv(f'{output_path}hipprightSR-lgm.csv')

    labels = {
        'leftSR':hippleftSRlabels,
        'rightSR':hipprightSRlabels,
        'leftOR': hippleftORlabels,
        'rightOR':hipprightORlabels,
    }
    return labels
