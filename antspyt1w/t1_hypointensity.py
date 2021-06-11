
def lesionFeatures(  inimgO, wmpriorIn, wmpriorskelIn):
  inimg = n4BiasFieldCorrection( inimgO,inimgO*0+1)
  inimg = rankIntensity( inimg )
  inimg = denoiseImage( inimg )
  imgreg = antsRegistration( bbt, inimg, 'AffineFast' )
  daB = deepAtropos( imgreg$warpedmovout, doPreprocessing = FALSE )
  wmprob = antsApplyTransforms(  inimg, daB$probabilityImages[[4]], imgreg$fwdtransforms, whichtoinvert=TRUE )
  reg = antsRegistration( inimg, template, 'SyN')
  wmskel = antsApplyTransforms( inimg, wmpriorskelIn, reg$fwdtransforms, interpolator='nearestNeighbor')
  wmprior = antsApplyTransforms( inimg, wmpriorIn, reg$fwdtransforms, interpolator='nearestNeighbor')
  wmwarped = antsApplyTransforms( wmpriorIn, wmprob, reg$invtransforms, interpolator='linear')

  realWM = thresholdImage( wmprior , 0.9, Inf )

  parcellateWMdnz = kmeansSegmentation( inimg, 2, realWM, verbose=T, mrf=0.3 )$probabilityimages[[1]]

  return {
    "wmprior":wmprior,
    "wmproborig":wmprob,
    "denoised":inimg,
    "kmeanswmorig":parcellateWMdnz,
    "aff":reg[fwdtransforms][1] }

template = ants.image_read( "~/data/white_matter_hypointensity/T_template0_BrainCerebellum.nii.gz")
wmprior = ants.image_read( "~/data/white_matter_hypointensity/antsBrainSegmentationPosteriors3.nii.gz")
wmpriorskel = skeletonize( thresholdImage( wmprior, 0.4, Inf ) )

bbt = ants.image_read( getANTsXNetData( "biobank" ) )
bbt = brainExtraction( bbt, "t1" ) * bbt
bbt = rankIntensity( bbt )

img1 = ants.image_read("~/data/white_matter_hypointensity/PPMI-3803-20120814-MRI_T1-I340756.nii.gz" )
img2 = ants.image_read("~/data/white_matter_hypointensity/PPMI_3118_MR_MPRAGE_GRAPPA2_br_raw_20160427125628761_82_S405061_I665282.nii")
ct=1
for ( imgx in list(img1,img2) ) {
oimg = antsImageClone( imgx )
img = reflectImage( imgx, axis=0 )
print(img)
bxt = brainExtraction( img, 't1combined[5]' ) %>% thresholdImage(2,3)
img = img * bxt

####################################################
slf = lesionFeatures( img, wmprior, wmpriorskel )

mybig = round(c(88*2,256,256)/2)
templatesmall = resampleImage( template, mybig, useVoxels=TRUE )
# features = rank+dnz-image, lprob, wprob, wprior at mybig resolution
f1 = as.array( antsApplyTransforms( templatesmall, slf$denoised, slf$aff, whichtoinvert=c(TRUE) ) )
f2 = as.array( antsApplyTransforms( templatesmall, slf$kmeanswmorig , slf$aff, whichtoinvert=c(TRUE) ) )
f3 = as.array( antsApplyTransforms( templatesmall, slf$wmprob , slf$aff, whichtoinvert=c(TRUE) ) )
f4 = as.array( antsApplyTransforms( templatesmall, slf$wmprior , slf$aff, whichtoinvert=c(TRUE) ) )
farr = abind::abind( f1,f2,f3,f4,along=4)
mdl =  createUnetModel3D( list(NULL,NULL,NULL,4),
  numberOfOutputs = 1,
  numberOfLayers = 4,
  mode = 'sigmoid' )
load_model_weights_hdf5( mdl, "unet.h5" )


pp = predict( mdl, array( farr, dim=c(1,dim(farr))))
refimg = slf$denoised %>% resampleImage( mybig, useVoxels=TRUE )
myles = as.antsImage( pp[1,,,,1] ) %>%
  antsCopyImageInfo2( templatesmall )
lesresam = antsApplyTransforms( slf$denoised, myles, slf$aff, whichtoinvert=c(FALSE) )

pdf( paste0( "wmh_example_",ct,".pdf" ), width=12, height=8 )
layout(matrix(1:2,nrow=2))
plot(slf$denoised,nslices=14,ncol=7,axis=3)
plot(slf$denoised,lesresam,nslices=14,ncol=7,axis=3)
dev.off()

# two pieces of evidence regarding whether there is a lesion or not
print("Max lesion prob:")
print(range(myles)) # this is important evidence of lesion presence
nchan=4
rnmdl =  createResNetModel3D( list(NULL,NULL,NULL,nchan),
  numberOfClassificationLabels = 1,
  layers = 1:3,
  residualBlockSchedule = c(3,4,6,3), squeezeAndExcite = TRUE,
  lowestResolution = 32, cardinality = 1, mode = "regression" )
load_model_weights_hdf5( rnmdl, 'discriminator.h5')
print("ResNetWMH-classifier: <0 = unlikely to have WMH lesion")
print( predict( rnmdl, array( farr, dim=c(1,dim(farr)))) )
ct=ct+1
}
