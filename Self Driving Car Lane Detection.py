#import useful libraries
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#upload test photo
test1 = plt.imread('test_samples/straight_lines1.jpg')
test11 = cv2.imread('test_samples/straight_lines1.jpg')
#cv2.imshow('test11',test11)
#cv2.waitKey(0)
#cv2.destroyAllWindows

#define some useful functions
def show_image(image, title = 'image', cmap_type = 'gray'):
    plt.imshow(image, cmap = cmap_type)
    plt.title(title)
    
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    
def get_M_Minv(sourcePints, destinationPoints):
  
    M = cv2.getPerspectiveTransform(sourcePints, destinationPoints)
    Minv = cv2.getPerspectiveTransform(destinationPoints, sourcePints)
    
    return M, Minv


def warp_image(img, M):

    # get the size of the image
    width = img.shape[1]
    height = img.shape[0]
    # Apply the transformation
    warped = cv2.warpPerspective(img, M, (width,height))
    
    return warped
    
    
def getSrcDstPoints(img):

    width = img.shape[1]
    height = img.shape[0]

    src1 = (width*2/5, height*2/3)
    src2 = (width*3/5, height*2/3)
    src3 = (width*5/6, height)
    src4 = (width/6, height)
    src = np.float32([src1, src2, src3, src4])

    dst1 = (320, 0)
    dst2 = (width - 320, 0)
    dst3 = (width - 320, height)
    dst4 = (320, height)
    dst = np.float32([dst1, dst2, dst3, dst4])
    
    return src , dst
    
    
# Functions for improving image details:

def mag_sobel_threshold(img , kernel = 3, thresh = (0, 255)):
    
    # get the image in gray
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Apply sobel edge detection in x and y directions
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = kernel)
    # Get the pixels magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2) 
    # Rescale the intensities
    scaled_magnitude = ((255 * magnitude) / np.max(magnitude)).astype(np.uint8) 
    # Create binary mask
    mag_binary = np.zeros_like(scaled_magnitude)
    # Combine the mask with the scaled magnitude (apply threshold)
    mag_binary[(scaled_magnitude >= thresh[0]) & (scaled_magnitude <= thresh[1])] = 1
    return mag_binary



def dir_sobel_threshold(img, kernel = 3, thresh = (0, np.pi/2)):
    
    # get the image in gray
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = kernel)
    # Take the absolute value
    abs_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create binary mask
    dir_binary =  np.zeros_like(abs_grad_dir)
    # Combine the mask with the scaled magnitude (apply threshold)
    dir_binary[(abs_dir >= thresh[0]) & (abs_dir <= thresh[1])] = 1
    return dir_binary


# Extract RGB and HLS and LAB color channels for better detection:

def get_rgb(img):
    #channel R has useful information
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]    
    return r, g, b

def get_hls(img):
    #channel l and s have useful information
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = img_HLS[:, :, 0]
    l = img_HLS[:, :, 1]
    s = img_HLS[:, :, 2]
    
    return h, l , s

def get_lab(img):
    #channel b has useful information
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l = img_LAB[:,:,0]
    a = img_LAB[:,:,1]
    b = img_LAB[:,:,2]
    
    return l, a, b
