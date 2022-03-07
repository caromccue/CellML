'''
Utility functions for image manipulation.
'''

from datetime import datetime
from PIL import Image

import cv2
import matplotlib
matplotlib.use('Qt4Agg', force=True)
from matplotlib.widgets import RectangleSelector
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import logging


def get_date_taken(path):
    '''Return the date image was taken from EXIF data'''
    return datetime.strptime(Image.open(path)._getexif()[36867], '%Y:%m:%d %H:%M:%S')

def open_grey_scale_image(path):
    '''Opens an image and converts it to ubyte and greyscale'''
    image = cv2.imread(path, 0)
    if image is None:
        raise OSError("File does not exist or is not an image: {}".format(path))
    return image

def clear_border(image):
    '''Removes connected (white) items from the border of an image.'''
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for i in range(h-1): # Iterate on the lines
        if image[i, 0] == 255:
            cv2.floodFill(image, mask, (0, i), 0)
        if image[i, w-1] == 255:
            cv2.floodFill(image, mask, (w-1, i), 0)
    for i in range(w-1): # Iterate on the columns
        if image[0, i] == 255:
            cv2.floodFill(image, mask, (i, 0), 0)
        if image[h-1, i] == 255:
            cv2.floodFill(image, mask, (i, h-1), 0)
    return image