from heatmap import*
from preprocessing import*
from BryantSequence import*
import os
import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
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


# #rename all folders for the first time
# i=0
# path='/mnt/DATA/cephalo/data_dicom_masks_annotations/'   #change according to your path
# os.chdir(path)
# for dir in os.listdir(path):
#     print(len(dir.split('.')))
#     if (len(dir.split('.'))==1):
#         i += 1
#         os.rename(str(dir), str('patient_%d' % i))

#get coordinates
origin = []
for i in range(1, 201):
    dicom_dir = 'patient_%d/dicom' % i
    o = get_origin(dicom_dir)
    origin.append(o)
origin = np.array(origin)

co = []
for i in range(1, 201):
    dicom_dir = 'patient_%d' % i
    c = load_coordinates(dicom_dir)
    co.append(c)
coordinates = np.array(co, dtype=float)

subtract = []
for i in range(200):
    t = np.tile(origin[i], (21, 1))
    subtract.append(t)
subtract = np.array(subtract)
aco = coordinates - subtract

extreme = []
for i in range(200):
    m = [np.min(aco[i, :, 0]), np.max(aco[i, :, 0]), np.min(aco[i, :, 1]), np.max(aco[i, :, 1]), np.min(aco[i, :, 2]),
         np.max(aco[i, :, 2])]
    extreme.append(m)
extreme = np.array(extreme)

cut = np.around(extreme[:, [0, 2, 4]] - 16)
cut[:, 0] = cut[:, 0] - 8
cut[cut < 0] = 0
cut = cut.astype(int)

for i in range(200):
    aco[i, :, 2] = aco[i, :, 2] - cut[i, 2]
    aco[i, :, 1] = aco[i, :, 1] - cut[i, 1]
    aco[i, :, 0] = aco[i, :, 0] - cut[i, 0]

extreme1 = []
for i in range(200):
    m = [np.min(aco[i, :, 0]), np.max(aco[i, :, 0]), np.min(aco[i, :, 1]), np.max(aco[i, :, 1]), np.min(aco[i, :, 2]),
         np.max(aco[i, :, 2])]
    extreme1.append(m)
extreme1 = np.array(extreme1)

max = np.max(extreme1[:, [1, 3, 5]], axis=0)
max = np.ceil(max / 8) * 8
max = max.astype(int)
max[0] = max[0] + 8
print('image size:', max)
np.save('z200cube.npy', aco)

#preprocessing (200 samples in 5 batch)
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('patient_number', cell_overwrite_ok=True)
for i in range(1, 6):  ##number of batches
    input = []
    seg = []
    for j in range(i * 40 - 39, i * 40 + 1):
        segment = []
        excel_dir = 'patient_%d' % j
        dicom_dir = 'patient_%d/dicom' % j
        segment_dir1 = 'patient_%d/masks/mandibula' % j
        segment_dir2 = 'patient_%d/masks/upper_skull' % j
        sheet.write(j - 1, 0, excel_dir)
        files = os.listdir(excel_dir)
        for s in files:
            if (os.path.splitext(s)[-1][1:] == "xlsx"):
                sheet.write(j - 1, 1, s[0:12])
                # read dicom file(dicom tags)
        slices = load_patient(dicom_dir)
        # get spacing of file
        pixel_spacing = slices[0].PixelSpacing
        pixel_spacing.append(slices[0].SliceThickness)
        pixel_spacing[2], pixel_spacing[0] = pixel_spacing[0], pixel_spacing[2]

        # extract gray values
        image = get_pixels_hu_by_simpleitk(dicom_dir)
        mandibula = read_bmp(segment_dir1)
        upperskull = read_bmp(segment_dir2)

        # normalize to 1/1/1
        image = resample(image, pixel_spacing)
        mandibula = resample(mandibula, pixel_spacing)
        upperskull = resample(upperskull, pixel_spacing)

        # pad
        upperskull = np.pad(upperskull,
                            ((0, 400 - image.shape[0]), (0, 400 - image.shape[1]), (0, 400 - upperskull.shape[2])),
                            'constant', constant_values=(0, 0))
        mandibula = np.pad(mandibula,
                           ((0, 400 - image.shape[0]), (0, 400 - image.shape[1]), (0, 400 - mandibula.shape[2])),
                           'constant', constant_values=(0, 0))
        image = np.pad(image, ((0, 400 - image.shape[0]), (0, 400 - image.shape[1]), (0, 400 - image.shape[2])),
                       'constant', constant_values=(0, 0))

        # crop
        upperskull = upperskull[cut[j - 1, 0]:cut[j - 1, 0] + max[0], cut[j - 1, 1]:cut[j - 1, 1] + max[1],
                     cut[j - 1, 2]:cut[j - 1, 2] + max[2]]
        mandibula = mandibula[cut[j - 1, 0]:cut[j - 1, 0] + max[0], cut[j - 1, 1]:cut[j - 1, 1] + max[1],
                    cut[j - 1, 2]:cut[j - 1, 2] + max[2]]
        image = image[cut[j - 1, 0]:cut[j - 1, 0] + max[0], cut[j - 1, 1]:cut[j - 1, 1] + max[1],
                cut[j - 1, 2]:cut[j - 1, 2] + max[2]]
        print('final shape:', image.shape)
        print('current sample: No.%d' % j)

        input.append(image)
        segment.append(mandibula)
        segment.append(upperskull)
        seg.append(segment)

    input = np.array(input)
    seg = np.array(seg)
    seg = seg.transpose((0, 2, 3, 4, 1))
    np.save('batch_%d.npy' % i, input)
    np.savez_compressed('batch_%d_seg.npz' % i, seg=seg)
book.save('correspondence.xls')

ct = []
seg = []
for i in range(1, 6):
    if (i == 1):
        ct = np.load('batch_%d.npy' % i)
        segmentation = np.load('batch_%d_seg.npz' % i)
        seg = segmentation['seg']
    else:
        a = np.load('batch_%d.npy' % i)
        b = np.load('batch_%d_seg.npz' % i)
        b = b['seg']
        ct = np.concatenate((ct, a), axis=0)
        seg = np.concatenate((seg, b), axis=0)
print('ct.shape:', ct.shape, 'seg.shape:', seg.shape)
np.savez_compressed('dataset200.npz', ct=ct, seg=seg)

hm=[]
for j in range(4):   #number of landmarks
  h=[]
  for i in range(200):  #number of patients
      xres = 128
      yres = 136
      zres = 144
      xlim = (0, xres)
      ylim = (0, yres)
      zlim = (0, zres)
      x = np.arange(xres, dtype=np.float)
      y = np.arange(yres, dtype=np.float)
      z = np.arange(zres, dtype=np.float)
      xx, yy,zz = np.meshgrid(x, y,z)

      # evaluate kernels at grid points
      xxyyzz = np.c_[xx.ravel(), yy.ravel(),zz.ravel()]

      img = draw_heatmap(xres,yres,zres,coordinates[i,j,1],coordinates[i,j,0],coordinates[i,j,2],4,xxyyzz.copy())
      h.append(img)
  hm.append(h)
hm=np.array(hm)
hm=hm.transpose((1,2,3,4,0))
np.save('3d_heatmap.npy', hm)


def get_BryantSequence(rotc):
    BS = []
    for i in range(35):
        #read 3 landmarks
        pl = np.array(rotc[i, 1, :])
        pr = np.array(rotc[i, 2, :])
        ol = np.array(rotc[i, 0, :])

        #vector calculation
        x = pl - pr
        x_norm = x / np.sqrt(np.sum(x ** 2))
        origin = pl - x * (pl[0] - ol[0]) / (pl[0] - pr[0])
        y = origin - ol
        y_norm = y / np.sqrt(np.sum(y ** 2))
        z_norm = np.cross(x_norm, y_norm)


        L = [Matrix(x_norm), Matrix(y_norm), Matrix(z_norm)]
        r = GramSchmidt(L, True)
        rotation = np.array(r)

        # in Radian
        a = math.atan(-rotation[1, 2] / rotation[2, 2])
        b = math.asin(rotation[0, 2]);
        c = math.atan(-rotation[0, 1] / rotation[0, 0])

        #  in °
        a = a * 180 / 3.1416
        b = b * 180 / 3.1416
        c = c * 180 / 3.1416
        bs = [a, b, c]
        BS.append(bs)
    BS = np.array(BS)
    return BS


def rot(bs, input):
    image = np.pad(input, ((10, 10), (10, 10), (10, 10)), 'constant', constant_values=(0, 0))
    image1 = scipy.ndimage.interpolation.rotate(image, bs[0], mode='nearest', axes=(1, 2), reshape=True)
    image2 = scipy.ndimage.interpolation.rotate(image1, bs[1], mode='nearest', axes=(0, 2), reshape=True)
    image3 = scipy.ndimage.interpolation.rotate(image2, bs[2], mode='nearest', axes=(0, 1), reshape=True)
    shape = [round(image3.shape[0] / 2), round(image3.shape[1] / 2), round(image3.shape[2] / 2)]
    image3 = image3[shape[0] - 84:shape[0] + 84, shape[1] - 76:shape[1] + 76, shape[2] - 80:shape[2] + 80]
    return image3


rotc = coordinates[:, [14, 18, 19], :].copy()
bs = get_BryantSequence(rotc)
np.save('bryant_sequence.npy', bs)










