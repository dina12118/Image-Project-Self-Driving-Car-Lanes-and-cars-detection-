# Self Driving Car Lane Detection Project

in this project our goal is to write software pipeline that identify the lane boundaries in a
video from a front-facing camera on a car.


Steps of this project are the following:

Use  RGB, LAB, HSL color transforms, sobel gradients with each agnitude and direction and canny edge detection to create a thresholded binary image.
Apply a perspective transform to the resulted binary image ("birds-eye view").
Using sliding window algorithm to detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output image/video with lane well detected
