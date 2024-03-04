import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from Store.ImageStorer import ImageStorer


class StereoDisparityService:
    def __init__(self, image_storer: ImageStorer):
        self.imageStorer = image_storer

    def resize_image(self, image, scale_factor):
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


    def showStereoDisparityMap(self):
        scale_factor = 0.25

        imgL = self.resize_image(self.imageStorer.load_imageL(),scale_factor)
        imgR = self.resize_image(self.imageStorer.load_imageR(),scale_factor)
        
        disparity = self.compute_disparity(imgL, imgR)
        disparity_display = cv.normalize(disparity, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        callback_params = {
            'image_right': imgR,
            'disparity': disparity,
            'scale_factor': scale_factor 
        }

        cv.namedWindow('Left Image')
        cv.setMouseCallback('Left Image', self.mouse_callback, callback_params)

        # Display images
        cv.imshow('Left Image', imgL)
        cv.imshow('Right Image', imgR)
        cv.imshow('Disparity Map', disparity_display)

        # Wait for the user to press 'q' to exit
        cv.waitKey()

        # Cleanup
        cv.destroyAllWindows()
    

    def compute_disparity(self, imgL, imgR):
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        stereo = cv.StereoBM.create(numDisparities=16*7,blockSize=11)
        stereo.setUniquenessRatio(10)
        disparity = stereo.compute(grayL, grayR).astype(np.float32)/16
        return disparity
    
    def mouse_callback(self, event, x, y, flags, param):
        imgR = param['image_right']
        disparity = param['disparity']
        scale_factor = param['scale_factor']
        if event == cv.EVENT_LBUTTONDOWN:
            # Calculate the corresponding coordinates in the right image
            disparity_value = disparity[y, x]

            if disparity_value == 0:
                print('Failure case')
                return
            
            x_right = int(x - disparity_value)
            if 0 <= x_right:
                imgR_with_circle = imgR.copy()
                cv.circle(imgR_with_circle, (x_right, y), 5, (0, 255, 0), -1)
                cv.imshow('Right Image', imgR_with_circle)
                print(f'({x},{y}),dis:{disparity_value}')
