import os
import SimpleITK as sitk
# import glob
import numpy as np


def read_dcm_to_array(file_dir, min_hu=-1150, max_hu=350, normalize=True):
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(filenames)
    sitk_image = reader.Execute()
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    if normalize:
        numpy_image = np.array(numpy_image, np.float32)
        numpy_image[numpy_image > max_hu] = max_hu
        numpy_image[numpy_image < min_hu] = min_hu
        numpy_image = (numpy_image - min_hu) / float(max_hu - min_hu) * 255
    return numpy_image


def normalize_image(image, min_hu=-1150, max_hu=350):
    image = np.array(image, np.float)
    image[image > max_hu] = max_hu
    image[image < min_hu] = min_hu
    image = (image - min_hu) / float(max_hu - min_hu) * 255
    return image

def save_image(ndimage, output_path, z_dimension=2):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if np.ndim(ndimage)==2:
        absolute_path=os.path.join(output_path,'1.png')
        scipy.misc.imsave(absolute_path,ndimage)
    elif np.ndim(ndimage)==3:
        for i in range(ndimage.shape[z_dimension]):
            if z_dimension==2:
                '''coordinate order must be y-x-z'''
                slice = ndimage[:, :, i]
            elif z_dimension==0:
                '''coordinate order must be z-y-x'''
                slice = ndimage[i, :, :]
            else:
                raise ValueError('incorrect image coordinate order')
            slice = slice.astype(np.float) / np.max(slice) * 255
            absolute_path = os.path.join(output_path, str(i+10001) + '.png')
            cv2.imwrite(absolute_path, slice)
            # scipy.misc.imsave(absolute_path, slice)

def save_images(train_data_folders, outdir):
    j = 0
    print 'saving images'
    for i in train_data_folders:
        j += 1
        if j % 50 == 0:
            print 'number of images saved : ', j
        images = read_dcm_to_array(i)
        path = i.split('/')[-1]
        outpath = os.path.join(outdir, path)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        save_image(read_dcm_to_array(i), outpath, z_dimension=0)
