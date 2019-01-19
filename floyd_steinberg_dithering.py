import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

def plot(img, mode=None):
    if mode is None:
        mode = cv2.COLOR_BGR2RGB
    plt.imshow(cv2.cvtColor(img, mode))
    plt.show()
    
def insert_simple(img_to, img_what, x1, y1, x2, y2):
    w = x2-x1
    h = y2-y1
    if img_what.shape[0] != h or img_what.shape[1] != w:
        img_what = cv2.resize(img_what,(w,h))
    img_to[y1:y2, x1:x2] = img_what

@jit
def get_range(x_min, x_max, x1_min, x1_max, cen, cen1):
    dx1 = cen1 - x1_min
    res_min = cen - dx1
    if res_min < x_min:
        res_min = x_min
    res_min = int(res_min)
    res_max = res_min + (x1_max - x1_min)
    if res_max > x_max:
        res_max = x_max
        res_min = int(res_max - (x1_max - x1_min))
    res_max = int(res_max)
    return res_min, res_max
        
@jit
def get_paste_diaps(img_large, img_paste, center_large, center_paste, \
                    offset_large_x=0, offset_large_y=0):
    y_max_large, x_max_large, nope = img_large.shape
    y_max_paste, x_max_paste, nope = img_paste.shape
    
    x_max_large -= offset_large_x
    x_min_large = offset_large_x
    y_max_large -= offset_large_y
    y_min_large = offset_large_y
    
    x_p_min, x_p_max = get_range(x_min_large, x_max_large, 0, x_max_paste, 
                                 center_large[0], center_paste[0])
    y_p_min, y_p_max = get_range(y_min_large, y_max_large, 0, y_max_paste, 
                                 center_large[1], center_paste[1])
    return x_p_min, x_p_max, y_p_min, y_p_max 
    
@jit
def paste_with_mask(img_large, img_paste, mask_paste, center_large, center_paste):
    x1, x2, y1, y2 = get_paste_diaps(img_large, img_paste, center_large, center_paste)
    for i in range(y1,y2):
        for j in range(x1,x2):
            if mask_paste[i-y1, j-x1, 0] != 0:
                img_large[i,j,:] = img_paste[i-y1, j-x1,:]

@jit
def get_good_pix(pix):
    if isinstance(pix, np.ndarray):
        for i in range(pix.shape[0]):
            if pix[i] > 255:
                pix[i] = 255
            elif pix[i] < 0:
                pix[i] = 0
        return pix.astype(np.uint8)
    return 254 if pix > 254 else (1 if pix < 1 else int(pix))

@jit
def find_closest_palette_color(pix, factor):
    return get_good_pix(np.round(factor * pix.astype(np.float)/255.0) *(255.0/factor))

@jit
def kernel_fsd(img, error, x, y, dx, dy, multy, img_w, img_h):
    if x + dx < 0 or x+dx >= img_w or y+dy >= img_h:
        return
    img[y+dy, x+dx] = get_good_pix(img[y+dy, x+dx].astype(np.float) + error.astype(np.float) * multy)

@jit
def floyd_steinberg_dithering(img, factor=2):
    pixel = img.copy()
    if len(pixel.shape) == 2:
        pixel = np.reshape(pixel, (pixel.shape[0], pixel.shape[1], 1))
    img_h, img_w, r = pixel.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            oldpixel = pixel[i,j].copy()
            newpixel = find_closest_palette_color(oldpixel, factor)
            pixel[i,j] = newpixel
            quant_error = get_good_pix(oldpixel.astype(np.float)  - newpixel.astype(np.float))  
            
            kernel_fsd(pixel, quant_error, j, i, 1, 0,  7/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 2, 0,  5/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, -2, 1, 3/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, -1, 1, 5/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 0,  1, 7/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 1,  1, 5/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 2,  1, 3/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, -2, 2, 1/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, -1, 2, 3/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 0,  2, 5/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 1,  2, 3/48, img_w, img_h)
            kernel_fsd(pixel, quant_error, j, i, 2,  2, 1/48, img_w, img_h)
    pixel = np.reshape(pixel, img.shape)
    return pixel
