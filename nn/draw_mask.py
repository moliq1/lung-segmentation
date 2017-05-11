import numpy as np
import cv2
from skimage.measure import regionprops, find_contours
import os

def insertPoints(im, location=None,color='red'):
    im = np.array(im, dtype='int32')
    size = im.shape
    if len(size) == 2:
        imtest = np.dstack((im, im, im))
    else:
        imtest = im
    color_number=-1
    if color=='red':
        color_number=0
    elif color=='green':
        color_number=1
    elif color=='blue':
        color_number=0

    for n in range(len(location)):
        imtest[int(location[n, 0]), int(location[n, 1]), :] = 0
        imtest[int(location[n, 0]), int(location[n, 1]), color_number] = 255
    return imtest


def plot_boundaries(im, mask, color='red'):
    contours = find_contours(mask, 0.5)
    locations = []
    for n, contour in enumerate(contours):
        im = insertPoints(im, contour,color)
    if len(im.shape) == 2:
        imtest = np.dstack((im, im, im))
    # props =regionprops(mask)
    return im


def save_boundary_images(original_image, masks,output_path):
    '''save mask boundary by slice into original image'''
    for single_mask in masks:
        assert (original_image.shape == single_mask.shape)
    for depth in range(original_image.shape[0]):
        orginal_silce = original_image[:, :, depth]
        for i,single_mask in enumerate(masks):
            if i==0:
                color='red'
            elif i==1:
                color='green'
            elif i==2:
                color='blue'
            else:
                raise ValueError('mask number cannot exceed 3')
            mask_slice = single_mask[:, :, depth]
            orginal_silce = plot_boundaries(orginal_silce, mask_slice,color)
        absolute_path = os.path.join(output_path, str(depth) + '.png')
        cv2.imwrite(absolute_path, orginal_silce)
