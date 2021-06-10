
import numpy as np
import ants
import tensorflow as tf
import antspynet
import tempfile
import warnings


def images_to_list( x ):
    """
    Convert a list of filenames to a list of images

    Arguments
    ---------
    x : list of filenames

    Returns
    -------
    list of ANTsImages

    """
    outlist = []
    for k in range(len(x)):
        outlist.append( ants.image_read( x[k] ) )
    return outlist


def list_to_string(s, separator='-'):
    """
    Convert a list of numeric types to a string with the same information.

    Arguments
    ---------
    s : list of integers

    separator : a placeholder between strings

    Returns
    -------
    string

    Example
    -------
    >>> list_to_string( [0,1], "-" )
    """
    str1 = ""
    for ele in s:
        str1 += (separator+str(ele))
    return str1


def check_for_labels_in_image( label_list, img ):
    """
    Apply a two-channel super resolution model to an image and probability pair.

    Arguments
    ---------
    label_list : list of integers

    img : ANTsImage of a segmentation

    Returns
    -------
    boolean

    Example
    -------
    >>> FIXME
    """
    imglabels = img.unique()
    isin = True
    for x in range( len( label_list ) ):
        isin = isin & ( label_list[x] in imglabels )
    return isin



def sort_library_by_similarity( img, img_segmentation,
    segmentation_numbers, library_intensity, library_segmentation,
    transformation='AffineFast' ):
    """
    Convert a list of numeric types to a string with the same information.

    Arguments
    ---------

    img : target image

    img_segmentation : target image segmentation containing the segmentation_numbers

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    library_intensity : list of strings or ANTsImages
        the list of library intensity images

    library_segmentation : list of strings or ANTsImages
        the list of library segmentation images

    transformation : type of transform used in ants.registration

    Returns
    -------
    dictionary:
        'sorted_library_int': sorted library images,
        'sorted_library_seg': sorted library segmentations,
        'sorted_similarity':  sorted similarity scores (lower better),
        'original_similarity': original similarity scores (lower better)

    Example
    -------
    import ants
    from superiq import sort_library_by_similarity
    targetimage = ants.image_read( ants.get_data("r16") )
    img0 = ants.add_noise_to_image( targetimage, "additivegaussian", ( 0, 2 ) )
    img1 = ants.image_read( ants.get_data("r27") )
    img2 = ants.image_read( ants.get_data("r16") ).add_noise_to_image( "additivegaussian", (0, 6) )
    tseg = ants.threshold_image( targetimage, "Otsu" , 3 )
    img0s = ants.threshold_image( img0, "Otsu" , 3 )
    img1s = ants.threshold_image( img1, "Otsu" , 3 )
    img2s = ants.threshold_image( img2, "Otsu" , 3 )
    ilist=[img2,img1,img0]
    slist=[img2s,img1s,img0s]
    ss = sort_library_by_similarity( targetimage, tseg, [3], ilist, slist )
    """
    if type(library_intensity[0]) == type(str(0)): # these are filenames
        libraryI = []
        for fn in library_intensity:
            libraryI.append( ants.iMath( ants.image_read(fn), "Normalize" ) )
        libraryL = []
        for fn in library_segmentation:
            temp = ants.image_read(fn)
#            temp = ants.mask_image( temp, temp, segmentation_numbers )
            if not check_for_labels_in_image( segmentation_numbers, temp ):
                warnings.warn( "segmentation_numbers do not exist in" + fn )
            libraryL.append( temp )
    else:
        libraryI = []
        libraryL = []
        for x in range( len( library_segmentation ) ):
            libraryI.append( ants.iMath( library_intensity[x], "Normalize" ) )
            temp = library_segmentation[x]
#            temp = ants.mask_image( temp, temp, segmentation_numbers )
            if not check_for_labels_in_image( segmentation_numbers, temp ):
                warnings.warn( "segmentation_numbers do not exist in" + fn )
            libraryL.append( temp )
    similarity = []
    tempbin = ants.mask_image( img_segmentation, img_segmentation, segmentation_numbers, binarize=True )
    tempbin = ants.iMath( tempbin, "MD", 6 )
    imgc = ants.crop_image( ants.iMath( img, "Normalize"), tempbin )
    for x in range( len( library_segmentation ) ):
        tempbinlib = ants.mask_image( libraryL[x],
            libraryL[x], segmentation_numbers, binarize=True )
        fastaff = ants.registration( tempbin, tempbinlib, transformation )['fwdtransforms']
        reg = ants.registration( imgc, libraryI[x], "SyNOnly", initial_transform=fastaff[0] )
        mysim = ants.image_mutual_information( imgc, reg['warpedmovout' ] )
        similarity.append( mysim )

    zipped_lists1 = zip(similarity, libraryI)
    zipped_lists1 = sorted(zipped_lists1)
    sorted_list1 = [element for _, element in zipped_lists1]


    zipped_lists2 = zip(similarity, libraryL)
    zipped_lists2 = sorted(zipped_lists2)
    sorted_list2 = [element for _, element in zipped_lists2]
    similarity_sort = similarity.copy()
    similarity_sort.sort()
    return {
    'sorted_library_int':sorted_list1,
    'sorted_library_seg':sorted_list2,
    'sorted_similarity':similarity_sort,
    'original_similarity':similarity,
    'ordering':sorted(range(len(similarity)), key=similarity.__getitem__)
    }

def super_resolution_segmentation_per_label(
    imgIn,
    segmentation, # segmentation label image
    upFactor,
    sr_model,
    segmentation_numbers,
    dilation_amount = 6,
    probability_images=None, # probability list
    probability_labels=None, # the segmentation ids for the probability image,
    max_lab_plus_one=True,
    verbose = False,
):
    """
    Apply a two-channel super resolution model to an image and probability pair.

    Arguments
    ---------
    imgIn : ANTsImage
        image to be upsampled

    segmentation : ANTsImage
        segmentation probability, n-ary or binary image

    upFactor : list
        the upsampling factors associated with the super-resolution model

    sr_model : tensorflow model or String
        for computing super-resolution; String denotes an interpolation type

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    dilation_amount : integer
        amount to pad around each segmentation defining region over which to
        computer the super-resolution in each label

    probability_images : list of ANTsImages
        list of probability images

    probability_labels : integer list
        providing the integer values associated with each probability image

    max_lab_plus_one : boolean
        add background label

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `super_resolution` : ANTsImage
            super_resolution image

        `super_resolution_segmentation` : ANTsImage
            super_resolution_segmentation image

        `segmentation_geometry` : list of data frame types
            segmentation geometry for each label

        `probability_images` : list of ANTsImage
            segmentation probability maps


    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """
    newspc = ( np.asarray( ants.get_spacing( imgIn ) ) ).tolist()
    for k in range(len(newspc)):
        newspc[k] = newspc[k]/upFactor[k]
    imgup = ants.resample_image( imgIn, newspc, use_voxels=False, interp_type=0 )
    imgsrfull = imgup * 0.0
    weightedavg = imgup * 0.0
    problist = []
    bkgdilate = 2
    segmentationUse = ants.image_clone( segmentation )
    segmentationUse = ants.mask_image( segmentationUse, segmentationUse, segmentation_numbers )
    segmentation_numbers_use = segmentation_numbers.copy()
    if max_lab_plus_one:
        background = ants.threshold_image( segmentationUse, 1, max(segmentation_numbers) )
        background = ants.iMath(background,"MD",bkgdilate) - background
        backgroundup = ants.resample_image_to_target( background, imgup, interp_type='linear' )
        segmentation_numbers_use.append( max(segmentation_numbers) + 1 )
        segmentationUse = segmentationUse + background * max(segmentation_numbers_use)

    for locallab in segmentation_numbers:
        if verbose:
            print( "SR-per-label:" + str( locallab ) )
        binseg = ants.threshold_image( segmentationUse, locallab, locallab )
        sizethresh = 2
        if ( binseg == 1 ).sum() < sizethresh :
            warnings.warn( "SR-per-label:" + str( locallab ) + ' is small' )
        # FIXME replace binseg with probimg and use minprob to threshold it after SR
        if ( binseg == 1 ).sum() >= sizethresh :
            minprob="NA"
            maxprob="NA"
            if probability_images is not None:
                whichprob = probability_labels.index(locallab)
                probimg = probability_images[whichprob].resample_image_to_target( binseg )
                minprob = min( probimg[ binseg >= 0.5 ] )
                maxprob = max( probimg[ binseg >= 0.5 ] )
            if verbose:
                print( "SR-per-label:" + str( locallab ) + " min/max-prob: " + str(minprob)+ " / " + str(maxprob)  )
            binsegdil = ants.iMath( ants.threshold_image( segmentationUse, locallab, locallab ), "MD", dilation_amount )
            binsegdil2input = ants.resample_image_to_target( binsegdil, imgIn, interp_type='nearestNeighbor'  )
            imgc = ants.crop_image( ants.iMath(imgIn,"Normalize"), binsegdil2input )
            imgc = imgc * 255 - 127.5 # for SR
            imgch = ants.crop_image( binseg, binsegdil )
            imgch = ants.iMath( imgch, "Normalize" ) * 255 - 127.5 # for SR
            if type( sr_model ) == type(""): # this is just for testing
                binsegup = ants.resample_image_to_target( binseg, imgup, interp_type='linear' )
                problist.append( binsegup )
            else:
                myarr = np.stack( [imgc.numpy(),imgch.numpy()],axis=3 )
                newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
                myarr = myarr.reshape( newshape )
                pred = sr_model.predict( myarr )
                imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
                imgsr = ants.copy_image_info( imgc, imgsr )
                newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
                ants.set_spacing( imgsr,  newspc )
                imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
                imgsrh = ants.copy_image_info( imgc, imgsrh )
                ants.set_spacing( imgsrh,  newspc )
                problist.append( imgsrh )
                imgsr = antspynet.regression_match_image( imgsr, ants.resample_image_to_target(imgup,imgsr) )
                contribtoavg = ants.resample_image_to_target( imgsr*0+1, imgup, interp_type='nearestNeighbor' )
                weightedavg = weightedavg + contribtoavg
                imgsrfull = imgsrfull + ants.resample_image_to_target( imgsr, imgup, interp_type='nearestNeighbor' )

    if max_lab_plus_one:
        problist.append( backgroundup )

    imgsrfull2 = imgsrfull
    selector = imgsrfull == 0
    imgsrfull2[ selector  ] = imgup[ selector ]
    weightedavg[ weightedavg == 0.0 ] = 1.0
    imgsrfull2=imgsrfull2/weightedavg
    imgsrfull2[ imgup == 0 ] = 0

    for k in range(len(problist)):
        problist[k] = ants.resample_image_to_target(problist[k],imgsrfull2,interp_type='linear')

    if max_lab_plus_one:
        tarmask = ants.threshold_image( segmentationUse, 1, segmentationUse.max() )
    else:
        tarmask = ants.threshold_image( segmentationUse, 1, segmentationUse.max() ).iMath("MD",1)
    tarmask = ants.resample_image_to_target( tarmask, imgsrfull2, interp_type='genericLabel' )
    segmat = ants.images_to_matrix(problist, tarmask)
    finalsegvec = segmat.argmax(axis=0)
    finalsegvec2 = finalsegvec.copy()

    # mapfinalsegvec to original labels
    for i in range(len(problist)):
        segnum = segmentation_numbers_use[i]
        finalsegvec2[finalsegvec == i] = segnum

    outimg = ants.make_image(tarmask, finalsegvec2)
    outimg = ants.mask_image( outimg, outimg, segmentation_numbers )
    seggeom = ants.label_geometry_measures( outimg )

    return {
        "super_resolution": imgsrfull2,
        "super_resolution_segmentation": outimg,
        "segmentation_geometry": seggeom,
        "probability_images": problist
        }



def ljlf_parcellation(
    img,
    segmentation_numbers,
    forward_transforms,
    template,
    templateLabels,
    library_intensity,
    library_segmentation,
    submask_dilation=12,  # a parameter that should be explored
    searcher=1,  # double this for SR
    radder=2,  # double this for SR
    reg_iterations = [100,100,5],
    syn_sampling=2,
    syn_metric='CC',
    max_lab_plus_one=False,
    localtx = "Affine",
    output_prefix=None,
    verbose=False,
):
    """
    Apply local joint label fusion to an image given a library.

    Arguments
    ---------
    img : ANTsImage
        image to be labeled

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    forward_transforms : list
        transformations that map the template labels to the img

    template : ANTsImages
        a reference template that provides the initial labeling.  could be a
        template from the library or a population-specific template

    templateLabels : ANTsImages
        the reference template segmentation image.

    library_intensity : list of strings or ANTsImages
        the list of library intensity images

    library_segmentation : list of strings or ANTsImages
        the list of library segmentation images

    submask_dilation : integer dilation of mask
        morphological operation that increases the size of the region of interest
        for registration and segmentation

    searcher :  integer search region
        see joint label fusion; this controls the search region

    radder :  integer
        controls the patch radius for similarity calculations

    reg_iterations : list of integers
        controlling the registration iterations; see ants.registration

    syn_sampling : integer
        the metric parameter for registration 2 for CC and 32 or 16 for mattes

    syn_metric : string
        the metric type usually CC or mattes

    max_lab_plus_one : boolean
        set True if you are having problems with background segmentation labels

    localtx : string
        type of local transformation

    output_prefix : string
        the location of the output; should be both a directory and prefix filename

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `ljlf` : key/value
            the local JLF object

        `segmentation` : ANTsImage
            the output segmentation image

    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """

    if output_prefix is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_prefix = str(temp_dir.name) + "/LJLF_"
        if verbose:
            print("Created temporary output location: " + output_prefix )


    # build the filenames
    ################################################################################
    if type(library_intensity[0]) == type(str(0)): # these are filenames
        libraryI = []
        for fn in library_intensity:
            libraryI.append( ants.iMath( ants.image_read(fn), "Normalize" ) )
        libraryL = []
        for fn in library_segmentation:
            temp = ants.image_read(fn)
#            temp = ants.mask_image( temp, temp, segmentation_numbers )
            if not check_for_labels_in_image( segmentation_numbers, temp ):
                warnings.warn( "segmentation_numbers do not exist in" + fn )
            libraryL.append( temp )
    else:
        libraryI = []
        libraryL = []
        for x in range( len( library_segmentation ) ):
            libraryI.append( ants.iMath( library_intensity[x], "Normalize" ) )
            temp = library_segmentation[x]
#            temp = ants.mask_image( temp, temp, segmentation_numbers )
            if not check_for_labels_in_image( segmentation_numbers, temp ):
                warnings.warn( "segmentation_numbers do not exist")
            libraryL.append( temp )

    ################################################################################
    if not check_for_labels_in_image( segmentation_numbers, templateLabels ):
        warnings.warn( "segmentation_numbers do not exist in templateLabels" )
    initlab0 = ants.apply_transforms(
        img, templateLabels, forward_transforms, interpolator="genericLabel"
    )
    initlab0 = ants.mask_image(initlab0, initlab0, segmentation_numbers)
    ################################################################################
    initlab = initlab0
    # get rid of cerebellum and brain stem
    ################################################################################
    # check outputs at this stage
    ################################################################################
    initlabThresh = ants.threshold_image(initlab, 1, 1e9)
    ################################################################################
    cropmask = ants.morphology(initlabThresh, "dilate", submask_dilation)
    imgc = ants.crop_image( ants.iMath( img, "Normalize"), cropmask)
    if verbose:
        print("Nerds want to know the size if dilation is:" + str(submask_dilation ))
        print( imgc )
    imgc = ants.iMath(imgc, "TruncateIntensity", 0.001, 0.99999)
    imgc = ants.iMath( imgc, "Normalize" )
    initlabc = ants.resample_image_to_target( initlab, imgc, interp_type="nearestNeighbor"  )
    jlfmask = imgc * 0 + 1
    deftx = "SyN"
    ljlf = ants.local_joint_label_fusion(
        target_image=imgc,
        which_labels=segmentation_numbers,
        target_mask=jlfmask,
        initial_label=initlabc,
        type_of_transform=deftx,  # FIXME - try SyN and SyNOnly
        submask_dilation=submask_dilation,  # we do it this way for consistency across SR and OR
        r_search=searcher,  # should explore 0, 1 and 2
        rad=radder,  # should keep 2 at low-res and search 2 to 4 at high-res
        atlas_list=libraryI,
        label_list=libraryL,
        local_mask_transform=localtx,
        reg_iterations=reg_iterations,
        syn_sampling=syn_sampling,
        syn_metric=syn_metric,
        beta=2,  # higher "sharper" more robust to outliers ( need to check this again )
        rho=0.1,
        nonnegative=True,
        max_lab_plus_one=max_lab_plus_one,
        verbose=verbose,
        output_prefix=output_prefix,
    )
    ################################################################################
    temp = ants.image_clone(ljlf["ljlf"]["segmentation"], pixeltype="float")
    temp = ants.mask_image( temp, temp, segmentation_numbers )
    hippLabelJLF = ants.resample_image_to_target( temp, img, interp_type="nearestNeighbor" )
    return {
        "ljlf": ljlf,
        "segmentation": hippLabelJLF,
    }




def ljlf_parcellation_one_template(
    img,
    segmentation_numbers,
    forward_transforms,
    template,
    templateLabels,
    templateRepeats,
    submask_dilation=6,  # a parameter that should be explored
    searcher=1,  # double this for SR
    radder=2,  # double this for SR
    reg_iterations = [100,100,100,55],
    syn_sampling=2,
    syn_metric='CC',
    max_lab_plus_one=True,
    deformation_sd=2.0,
    intensity_sd=0.1,
    output_prefix=None,
    verbose=False,
):
    """
    Apply local joint label fusion to an image given a library.

    Arguments
    ---------
    img : ANTsImage
        image to be labeled

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    forward_transforms : list
        transformations that map the template labels to the img

    template : ANTsImages
        a reference template that provides the initial labeling.  could be a
        template from the library or a population-specific template

    templateLabels : ANTsImages
        the reference template segmentation image.

    templateRepeats : integer number of registrations to perform
        repeats the template templateRepeats number of times to provide variability

    submask_dilation : integer dilation of mask
        morphological operation that increases the size of the region of interest
        for registration and segmentation

    searcher :  integer search region
        see joint label fusion; this controls the search region

    radder :  integer
        controls the patch radius for similarity calculations

    reg_iterations : list of integers
        controlling the registration iterations; see ants.registration

    syn_sampling : integer
        the metric parameter for registration 2 for CC and 32 or 16 for mattes

    syn_metric : string
        the metric type usually CC or mattes

    max_lab_plus_one : boolean
        set True if you are having problems with background segmentation labels

    deformation_sd : numeric value
        controls the amount of deformation in simulation

    intensity_sd : numeric value
        controls the amount of intensity noise in simulation

    output_prefix : string
        the location of the output; should be both a directory and prefix filename

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `ljlf` : key/value
            the local JLF object

        `segmentation` : ANTsImage
            the output segmentation image

    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """

    if output_prefix is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_prefix = str(temp_dir.name) + "/LJLF_"
        if verbose:
            print("Created temporary output location: " + output_prefix )

    # build the filenames
    ################################################################################
    libraryI = []
    libraryL = []
    for x in range(templateRepeats):
        temp = ants.iMath( template, "Normalize" )
        bsp_field = ants.simulate_displacement_field(template, field_type="bspline",sd_noise=deformation_sd)
        bsp_xfrm = ants.transform_from_displacement_field(bsp_field * 3)
        temp = ants.apply_ants_transform_to_image(bsp_xfrm, temp, temp)
        mystd = intensity_sd * temp.std()
        temp = ants.add_noise_to_image( temp, "additivegaussian", [0,mystd] )
        libraryI.append( temp )
        temp = ants.image_clone( templateLabels )
        temp = ants.mask_image( templateLabels, templateLabels, segmentation_numbers )
        temp = ants.apply_ants_transform_to_image(bsp_xfrm, temp, temp,interpolation='nearestneighbor')
        libraryL.append( temp )

    #  https://mindboggle.readthedocs.io/en/latest/labels.html
    ################################################################################
    initlab0 = ants.apply_transforms(
        img, templateLabels, forward_transforms, interpolator="genericLabel"
    )
    initlab = ants.mask_image(initlab0, initlab0, segmentation_numbers)
    ################################################################################
    if not check_for_labels_in_image( segmentation_numbers, templateLabels ):
        warnings.warn( "segmentation_numbers do not exist in templateLabels" )
    initlabThresh = ants.threshold_image(initlab, 1, 1e9)
    ################################################################################
    cropmask = ants.morphology(initlabThresh, "dilate", submask_dilation)
    imgc = ants.crop_image( ants.iMath( img, "Normalize"), cropmask)
    imgc = ants.iMath(imgc, "TruncateIntensity", 0.001, 0.99999)
    initlabc = ants.resample_image_to_target( initlab, imgc, interp_type="nearestNeighbor"  )
    jlfmask = imgc * 0 + 1
    deftx = "SyN"
    localtx = "Similarity"
    ljlf = ants.local_joint_label_fusion(
        target_image=imgc,
        which_labels=segmentation_numbers,
        target_mask=jlfmask,
        initial_label=initlabc,
        type_of_transform=deftx,  # FIXME - try SyN and SyNOnly
        submask_dilation=submask_dilation,  # we do it this way for consistency across SR and OR
        r_search=searcher,  # should explore 0, 1 and 2
        rad=radder,  # should keep 2 at low-res and search 2 to 4 at high-res
        atlas_list=libraryI,
        label_list=libraryL,
        local_mask_transform=localtx,
        reg_iterations=reg_iterations,
        syn_sampling=syn_sampling,
        syn_metric=syn_metric,
        beta=2,  # higher "sharper" more robust to outliers ( need to check this again )
        rho=0.1,
        nonnegative=True,
        max_lab_plus_one=max_lab_plus_one,
        verbose=verbose,
        output_prefix=output_prefix,
    )
    ################################################################################
    temp = ants.image_clone(ljlf["ljlf"]["segmentation"], pixeltype="float")
    temp = ants.mask_image( temp, temp, segmentation_numbers )
    hippLabelJLF = ants.resample_image_to_target( temp, img, interp_type="nearestNeighbor" )
    return {
        "ljlf": ljlf,
        "segmentation": hippLabelJLF,
    }
