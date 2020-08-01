from PIL import Image
import math
import cv2 as cv
import numpy as np

# Install Requires:
import skimage as ski

MM_PER_INCH = 25.4


def resize_img(img, resize_factor, output_filepath=None):
    """ Realiza un resize de la imagen con un factor de escala. Si el path de salida existe, guarda la misma a un archivo. """
    if resize_factor != 1:
        img = ski.transform.rescale(image=img, scale=resize_factor, multichannel=True, mode='constant', anti_aliasing=True)
        img = ski.img_as_ubyte(img)
    if bool(output_filepath):
        cv.imwrite(output_filepath, img)
        # ski.io.imsave(fname=filepath, arr=img)
    return img


def mm_to_px(mm, dpi=600, to_int=True):
    def mm2px(mm, dpi, to_int):
        px = (mm / MM_PER_INCH) * dpi
        if to_int:
            px = round(px)
        return px

    if isinstance(mm, tuple):
        px = tuple([mm2px(m, dpi, to_int) for m in mm])
    elif isinstance(mm, list):
        px = [mm2px(m, dpi, to_int) for m in mm]
    else:
        px = mm2px(mm, dpi, to_int)

    return px


def area_px(r_in_mm, dpi=600):
    return int(math.pi * mm_to_px(r_in_mm, dpi=dpi) ** 2)


def crop(image_path, output_path, hor_divisions=3, ver_divisions=4):
    im = Image.open(image_path)
    width, height = im.size
    for i in range(ver_divisions):
        for j in range(hor_divisions):
            cropped_image = im.crop((j*(width/hor_divisions), i*(height/ver_divisions), (j+1)*(width/hor_divisions), (i+1)*(height/ver_divisions)))
            cropped_image.save(output_path+'cropped_'+str(i+1)+'_'+str(j+1)+'.png')


def scale_img(img: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    This function enlarges or shrinks the image symmetrically.
    Args:
        img: image to be modified.
        scale_percent: OUT/IN size ratio.

    Returns:
        Modified image.
    """
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized


def mean_channels(array):
    return [array[..., i].mean() for i in range(array.shape[-1])]


def subtract_mean(array):
    m = mean_channels(array)
    return array - m

# from PyImageSearch:
def adjust_gamma(image: np.ndarray, gamma: int = 1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

