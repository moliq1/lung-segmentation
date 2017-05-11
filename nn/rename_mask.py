import os
import glob
import numpy as np
import SimpleITK as sitk
import cv2

def rename_all_mask():
    cases = os.listdir('/mnt/sdk/mask/')
    for case in cases:
        if '100001.png' in os.listdir('/mnt/sdk/mask/' + case):
            files = glob.glob('/mnt/sdk/mask/' + case + '/*')
            i = 0
            for f in files:
                i += 1
                print i
                file_dir = '/'.join(f.split('/')[:-1])
                filename = f.split('/')[-1]
                num = int(filename.split('.')[0]) - 90000
                new_name = file_dir + '/' + str(num) + '.png'
                os.rename(f, new_name)

def rename():
    case = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
    if '0.png' in os.listdir('/mnt/sdk/' + case):
        files = glob.glob('/mnt/sdk/' + case + '/*')

        for f in files:

            file_dir = '/'.join(f.split('/')[:-1])
            filename = f.split('/')[-1]
            num = int(filename.split('.')[0]) + 10001
            new_name = file_dir + '/' + str(num) + '.png'
            os.rename(f, new_name)


def get_rest_train():
    label_case = os.listdir('/mnt/sdk/mask')
    data_case = os.listdir('/mnt/sdk/lung_segmentation_2d_training_data')
    rest_case = [i for i in label_case if i not in data_case]
    print rest_case
    rest_case_check = [i for i in data_case if i not in label_case]
    print 'check case', rest_case_check
    return rest_case

def read_dcm_to_array(file_dir, min_hu=-1150, max_hu=350, normalize=True):
    '''
    functions to read dicom data
    '''
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(filenames)
    sitk_image = reader.Execute()
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    # resize pixel value between [0, 1]
    if normalize:
        numpy_image = np.array(numpy_image, np.float32)
        numpy_image[numpy_image > max_hu] = max_hu
        numpy_image[numpy_image < min_hu] = min_hu
        numpy_image = (numpy_image - min_hu) / float(max_hu - min_hu) * 255
    return numpy_image, sitk_image

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

    total_paths = glob.glob('/mnt/sdk/LIDC-IDRI/*/*/*')
    for case in train_data_folders:
        path = [path for path in total_paths if path.split('/')[-1] == case][0]
        images, _ = read_dcm_to_array(path)
        outpath = os.path.join(outdir, case)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        save_image(images, outpath, z_dimension=0)


if __name__ == '__main__':
    rename_all_mask()
    # rest_case = get_rest_train()
    # save_images(rest_case, '/mnt/sdk/')