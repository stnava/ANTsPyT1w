
def lesionFeatures(  inimgO, wmpriorIn, wmpriorskelIn):
  inimg = n4BiasFieldCorrection( inimgO,inimgO*0+1)
  inimg = rankIntensity( inimg )
  inimg = denoiseImage( inimg )
  imgreg = antsRegistration( bbt, inimg, 'AffineFast' )
  daB = deepAtropos( imgreg['warpedmovout'], doPreprocessing = FALSE )
  wmprob = antsApplyTransforms(  inimg, daB['probabilityImages'][3], imgreg['fwdtransforms'], whichtoinvert=TRUE )
  reg = antsRegistration( inimg, template, 'SyN')
  wmskel = antsApplyTransforms( inimg, wmpriorskelIn, reg['fwdtransforms'], interpolator='nearestNeighbor')
  wmprior = antsApplyTransforms( inimg, wmpriorIn, reg['fwdtransforms'], interpolator='nearestNeighbor')
  wmwarped = antsApplyTransforms( wmpriorIn, wmprob, reg['invtransforms'], interpolator='linear')

  realWM = thresholdImage( wmprior , 0.9, Inf )

  parcellateWMdnz = kmeansSegmentation( inimg, 2, realWM, verbose=T, mrf=0.3 )['probabilityimages'][0]

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

oimg = antsImageClone( imgx )
img = reflectImage( imgx, axis=0 )
print(img)
bxt = antspynet.brainExtraction( img, 't1combined[5]' ).thresholdImage(2,3)
img = img * bxt

####################################################
slf = lesionFeatures( img, wmprior, wmpriorskel )
