import os
import SimpleITK as sitk
import pydicom
import numpy as np
import glob
from tqdm import tqdm
import scipy.ndimage
import math
from scipy.linalg import*
from sympy import*
import scipy
import xlrd
import xlwt
from scipy.ndimage.interpolation import zoom
from scipy import sparse


def is_dicom_file(filename):
    '''
    :param filename: path of dicom file
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def get_origin(src_dir):
    '''
    :param src_dir: path of dicom file
    :return: image origin
    '''
    files = sorted(os.listdir(src_dir))
    i = 0
    for s in files:
        if (i == 0):
            if (os.path.splitext(s)[-1][1:] == "dcm"):
                ds = pydicom.read_file(src_dir + '/' + s)
                i = i + 1
                origin = np.array(ds.ImagePositionPatient)
    return origin


def load_coordinates(src_dir):
    '''
    :param src_dir: path of xlsx file
    :return: coordinates of landmarks
    '''
    files = os.listdir(src_dir)
    for s in files:
        if (os.path.splitext(s)[-1][1:] == "xlsx"):
            workbook = xlrd.open_workbook(src_dir + '/' + s)
            booksheet = workbook.sheet_by_index(0)
            list = []
            for i in range(12, 33):
                row = booksheet.row_values(i)
                list.append(row[1:4])
    coordinates = np.array(list)
    return coordinates


def load_patient(src_dir):
    '''
    :param src_dir: path of dicom file
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
     read all dicom file in the folder and extract grayvalue within (-4000 ~ 4000)
    :param src_dir: dicom path
    :return: image array
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    img_array[img_array == -2000]
    img_array = normalize_hu(img_array)
    return img_array


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    '''
    resample image to new resolution
    :param src_dir: image array, old&new spacing
    :return: image array
    '''

    # calcluate factor
    old_spacing = np.array(old_spacing)
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    image = image.transpose((2, 1, 0))
    print("Shape after: ", image.shape)
    return image


def normalize(image):
    'normalize grayvalue of image to [0,1]'
    MIN_BOUND = np.max(image)
    MAX_BOUND = np.min(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 0.5] = 1.
    image[image < 0.5] = 0.
    return image


def normalize_hu(image):
    '''
    normalize gray value from(-1000 ~ 3000) to (0~1)
    :param image
    :return: list image normalized
    '''
    if (np.min(image) > -500):
        MIN_BOUND = 0.0
        MAX_BOUND = 4000.0
    else:
        MIN_BOUND = -1000.0
        MAX_BOUND = 3000.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def load_patient_images(src_dir, wildcard="*.*", exclude_wildcards=[]):
    '''
    load all png images of a patients
    :param image
    :return: 3D array
    '''
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1,) + im.shape) for im in images]
    res = np.vstack(images)
    return res


def read_bmp(src_dir):
    '''
    read all segmentation map in bmp format
    :param src_dir: path of bmp file
    :return: image array
    '''
    files = sorted(os.listdir(src_dir))
    file_list = []
    for s in files:
        img = cv2.imread(src_dir + '/' + s, 0)
        file_list.append(img)
    res = np.array(file_list)
    return res


def save_cube_img(target_path, cube_img, rows, cols):
    '''
        save 3D image as 2D slices
        :param 2D path,3D input
        :return: 2D images
    '''
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)










