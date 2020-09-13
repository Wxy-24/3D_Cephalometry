import numpy as np
import math
import scipy.ndimage


#Bryant Sequence calculation
def get_BryantSequence(rotc):
    BS = []
    for i in range(35):
        pl = np.array(rotc[i, 1, :])
        pr = np.array(rotc[i, 2, :])
        ol = np.array(rotc[i, 0, :])

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











