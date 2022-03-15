'''
Methods to segment individual cells from an image of adherent cells on posts
'''

import os
import warnings
import logging
import numpy as np

from skimage import io, morphology, exposure
from skimage.color import rgb2gray, label2rgb
from skimage.feature import canny, blob_dog, blob_log, blob_doh,  peak_local_max
from skimage.filters import sobel, threshold_otsu, try_all_threshold, threshold_local, threshold_minimum
from skimage.segmentation import clear_border, watershed, random_walker
from skimage.measure import label, regionprops
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import closing

import cv2


import os


from tqdm import tqdm


from src.data.utils import open_grey_scale_image

def segment(img, postsize = 275, exp_clip_limit=30):
    '''
    Segments cells in an image using a watershed algorithm. OpenCV implementation.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of an image cropped to show only the droplets region
    exp_clip_limit: float [0-1], optional
        clip_limit parameter for adaptive equalization

    Returns
    -------
    (labeled: numpy.ndarray, num_regions: int)
        labeled: labeled array of the same shape as input image where each region is assigned a disctinct integer label.
        num_regions: number of labeled regions
    '''

    # Adaptive Equalization
    clahe = cv2.createCLAHE(clipLimit=exp_clip_limit, tileGridSize=(10,10))
    img_adapteq = clahe.apply(img)

    # Thresholding (OTSU)
    blur = cv2.GaussianBlur(img_adapteq, (3,3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove small dark regions
    remove_posts = morphology.remove_small_objects(binary, postsize)
    remove_posts = morphology.remove_small_holes(remove_posts, postsize)
    remove_posts = remove_posts.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    closed = cv2.morphologyEx(remove_posts, cv2.MORPH_CLOSE, kernel, iterations = 1)
    #fill_holes = ndi.morphology.binary_fill_holes(closed, structure=np.ones((3, 3))).astype('uint8')

    # noise removal
    kernel = np.ones((2,2),np.uint8)
    #opening = cv2.morphologyEx(closed,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations = 2)
    # sure background area
    #sure_bg = cv2.dilate(opening,kernel,iterations=3)
    sure_bg = cv2.dilate(closing,kernel,iterations=2)
    # Finding sure foreground area
    #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    #markers = markers+1
    # Now, mark the region of unknown with zero
    #markers[unknown > 0] = 0

    # Run the watershed algorithm
    three_channels = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    segmented = cv2.watershed(three_channels.astype('uint8'), markers)

    return (segmented, segmented.max()-1)


def extract_indiv_cells(img, labeled, border=30, area_upper_cutoff=3, area_lower_cutoff=2, ecc_cutoff=0.25):
    '''
    Separate the individual cells as their own image.

    Parameters
    ----------
    img: numpy.ndarray
        Array representing the greyscale values (0-255) of the segmented image.
    labeled: numpy.ndarray
        Label array corresponding to 'img' where each region is assigned a disctinct integer value
    border: int, optional
        Number of pixels to add on each side of the labeled area to produce the final image.
    ecc_cutoff: float, optional
        Maximum eccentricity value of the labeled region. Regions with higher eccentricity will be ignored.
    area_perc_cutoff: float, optional
        Minimum area as a percentage of the mean area

    Returns
    -------
    list(numpy.ndarray)
        list where each array corresponds to one of the labeled regions bounding box + the border region
    list(RegionProperties)
        regionProperties of the labeled regions
    '''

    # Get region props
    reg = regionprops(labeled, coordinates='rc')[1:] # First label corresponds to the background (OpenCV)

    # Initialize list of images
    img_list = []

    # Get original image size
    max_col = img.shape[1]
    max_row = img.shape[0]

    # Get area cutoff
    area_cutoff_upper = area_upper_cutoff * np.mean([region.area for region in reg])
    area_cutoff_lower = area_lower_cutoff * np.median([region.area for region in reg])

    reg_clean = [region for region in reg if region.area < area_cutoff_upper and region.area > area_cutoff_lower and region.eccentricity > ecc_cutoff]

    for region in reg_clean:
        (min_row, min_col, max_row, max_col) = region.bbox
        cell_image = img[np.max([min_row-border,0]):np.min([max_row+border,max_row]),np.max([min_col-border,0]):np.min([max_col+border,max_col])]
        contrast_stretch = exposure.rescale_intensity(cell_image, in_range=(0,255))
        #resized = cell_image * 255
        img_list.append(contrast_stretch)

    return img_list, reg_clean

def segment_cells_to_file(image_filename, save_overlay=False):

    if os.path.isdir(image_filename):
        img_list = [os.path.join(image_filename,f) for f in os.listdir(image_filename) if f.endswith('.jpg')]
    elif os.path.isfile(image_filename):
        img_list = [image_filename]

    for image_file in tqdm(img_list):
        # Open image
        image = open_grey_scale_image(image_file)

        # Segment image
        (labeled, num_regions) = segment(image)

        # Save the overlay image if requested
        if save_overlay:
            image_overlay = label2rgb(labeled, bg_label=0)
            filename = image_file.split('.')[0] + '_segmented.jpg'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename, image_overlay)

        # Extract individual cells
        cell_images, _ = extract_indiv_cells(image, labeled)

        # Output folder has the same name as the image by default
        out_directory = image_file.split('.')[0]

        if not os.path.exists(out_directory):
            os.mkdir(out_directory)

        logging.info("Saving segmented cells to %s", out_directory)

        # Save all the images in the output directory
        for (i, img) in enumerate(cell_images):
            name = os.path.join(out_directory, os.path.basename(image_file).split('.')[0] + '_cell_' + str(i) + '.jpg')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(name, img, check_contrast=False)
