from scipy.spatial import distance as dist

import torch
import numpy as np
import cv2

def get_histogram(img, mask):
    '''
    input:
        img is 4d pytorch tensor image
        mask is 2d np array mask
    output:
        flatten normalized color histogram        
    '''
    # convert from 4d pytorch (1, C, H, W) ~ [0, 1] to numpy array (C, H, W) ~ [0, 255]
    img_3d = img.view(img.shape[1:]).numpy() * 255
    
    # transpose image from (C, H, W) to (H, W, C)
    img_3d = np.transpose(img_3d, (1, 2, 0))
    
    # get histogram and normalize it
    hist = cv2.calcHist([img_3d], [0, 1, 2], mask.astype(np.uint8) * 255, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def color_sim(img_a, mask_a, img_b, mask_b, method='Correlation'):
    hist_a = get_histogram(img_a, mask_a)
    hist_b = get_histogram(img_b, mask_b)

    # 'Chi-Squared': cv2.HISTCMP_CHISQR, 'Hellinger': cv2.HISTCMP_BHATTACHARYYA
    methods = {
        'Correlation': cv2.HISTCMP_CORREL,
        'Intersection': cv2.HISTCMP_INTERSECT        
    }
        
    m = methods[method]
    if m is None:
        ValueError('Invalid name for comparison method.')
    
    return max(0, cv2.compareHist(hist_a, hist_b, m))