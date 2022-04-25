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

def abs_sobel_thresh(img, axes = 'x', kernel = 3, thresh = (0, 255)):
    
    # get the image in gray
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if(axes == 'x'):
        # Apply sobel edge detection in x direction
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = kernel)
    else:
        # Apply sobel edge detection in y direction
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = kernel)
    # Take the absolute value
    abs_sobel = np.absolute(sobel)
    # Rescale the intensities
    scaled_sobel = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8) 
    # Create binary mask
    grad_binary = np.zeros_like(scaled_sobel)
    # Combine the mask with the x or y gradient (apply threshold)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
 
    return grad_binary

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
    # Apply sobel edge detection in x and y directions
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = kernel)
    # Take the absolute value
    abs_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create binary mask
    dir_binary =  np.zeros_like(abs_dir)
    # Combine the mask with the absolute direction value (apply threshold)
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


def combined_color_channels_threshod(img, r_thresh=(225,255), l_thresh=(215,255), s_thresh=(170,255), b_thresh=(180,255)):

    _, l, s = get_hls(img)
    r, _, _ = get_rgb(img)
    _, _, b = get_lab(img)
    # Create binary mask
    color_binary = np.zeros_like(r)
    # Combine the mask with the desired channels (R, S, L, B) (apply threshold)
    color_binary[((r > r_thresh[0]) & (r <= r_thresh[1])) |
                 ((l > l_thresh[0]) & (l <= l_thresh[1])) | 
                 ((s > s_thresh[0]) & (s <= s_thresh[1])) |
                 ((b > b_thresh[0]) & (b <= b_thresh[1])) ] = 1
    
    return color_binary   


def combined_grad_color_threshold(img,grad_kernel = 3, gradx_thresh = (20,100), grady_thresh = (50,100),mag_kernel = 5, mag_thresh = (50,100),dir_kernel = 9, dir_thresh = (0.7,1.3)):
    
    color_binary = combined_color_channels_threshod(img)
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Apply sobel edge detection in x and y directions
    gradx = abs_sobel_thresh(img, 'x', grad_kernel, gradx_thresh)
    grady = abs_sobel_thresh(img, 'y', grad_kernel, grady_thresh)
    # get mag_sobel_threshold
    mag_binary = mag_sobel_threshold(img, mag_kernel, mag_thresh)
    # get dir_sobel_threshold
    dir_binary = dir_sobel_threshold(img, dir_kernel, dir_thresh)
    # Combined color channels and gradient in y and x and it directions and magnitudes (apply threshold)
    combined_binary = np.zeros_like(color_binary)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1
    
    return combined_binary

# Apply Canny Edge Detection
def auto_canny_threshould(image, sigma=0.33):
    
    # Apply bluring to iliminate the noise
    img_blured = gaussian_blur(image, 7)
    # compute the median of the single channel pixel intensities
    v = np.median(img_blured)
    # apply automatic Canny edge detection using the computed median
    lower_threshold = int(max(0, (1.0 - sigma) * v))
    upper_threshold = int(min(255, (1.0 + sigma) * v))
    # apply canny edge detection
    canny_edged = cv2.Canny(img_blured, lower_threshold, upper_threshold)
    # return the edged image
    return canny_edged

# combine all together to get binary image with good edge detection:

def combined_canny_grad_color_threshold(img):

    color_grad_binary = combined_grad_color_threshold(img)
    canny_edged = auto_canny_threshould(img)
    combined_binary = np.zeros_like(color_grad_binary)
    combined_binary[ (canny_edged == 1) | (color_grad_binary == 1)] = 1
    
    return combined_binary


# Apply warp Perspective to the binary image after process it
def get_binary_warped(img, M):

    combined_img = combined_canny_grad_color_threshold(img)   
    # Apply transformation
    warped = warp_image(combined_img, M)

    return warped
