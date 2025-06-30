import antspyt1w
import ants
# localsyn(img, template, hemiS, templateHemi, whichHemi, padder, iterations, output_prefix, total_sigma=0.5)

fn = antspyt1w.get_data('PPMI-3803-20120814-MRI_T1-I340756', target_extension='.nii.gz' )
fn2 = antspyt1w.get_data('28386-00000000-T1w-01', target_extension='.nii.gz' )
img= ants.image_read( fn )
img2= ants.image_read( fn2 )
msk=ants.get_mask(img)
msk2=ants.get_mask(img2)
its=[40,0,0,0]
# reproducibility
#
# region_reg(input_image, input_image_tissue_segmentation, input_image_region_segmentation, input_template, input_template_region_segmentation, output_prefix, padding=10, labels_to_register=[2, 3, 4, 5], total_sigma=0.5, is_test=False)
lsyn0=antspyt1w.region_reg(img, msk, msk, img2, msk2,  padding=8,  output_prefix='/tmp/BB', total_sigma=0.5, is_test=True )
lsyn1=antspyt1w.region_reg(img, msk, msk, img2, msk2,  padding=8,  output_prefix='/tmp/CC', total_sigma=0.5, is_test=True )
# lsyn0=antspyt1w.localsyn(img, img2, msk, msk2, whichHemi=1, padder=0, iterations=its, output_prefix='/tmp/BB', total_sigma=0.5)
# lsyn1=antspyt1w.localsyn(img, img2, msk, msk2, whichHemi=1, padder=0, iterations=its, output_prefix='/tmp/CC', total_sigma=0.5)
