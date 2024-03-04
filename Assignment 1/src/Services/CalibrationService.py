import cv2 as cv
import numpy as np

from Store.ImageStorer import ImageStorer


class CalibrationService:
    def __init__(self, image_storer:ImageStorer):
        self.imageStorer = image_storer

    def find_corners(self):
        images = self.imageStorer.load_images()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for img in images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (11, 8), None)

            if found == True:
                refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                cv.drawChessboardCorners(img, (11, 8), refinedCorners, found)

                cv.namedWindow('img', 0)
                cv.resizeWindow('img', 1000, 1000)
                cv.imshow('img', img)
                cv.waitKey(300)
        cv.destroyAllWindows()

    def find_intrinsic(self):
        images = self.imageStorer.load_images()
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []
        imgpoints = [] 

        for img in images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (11, 8), None)

            if found == True:
                objpoints.append(objp)
                refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(refinedCorners)
        
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Intrinsic:")
        print(mtx)

    def find_extrinsic(self,filename:str):
        img = self.imageStorer.load_image_by_name(filename)
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []
        imgpoints = [] 

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, (11, 8), None)

        if found:
            objpoints.append(objp)
            refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(refinedCorners)
            ret, mtx, dist, rvecs, tvecs  = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            R, _ = cv.Rodrigues(rvecs[0])
            extrinsic_matrix = np.hstack((R, tvecs[0]))
            print("Extrinsic:")
            print(extrinsic_matrix)

    def find_distortion(self):
            images = self.imageStorer.load_images()
            objp = np.zeros((11*8, 3), np.float32)
            objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            objpoints = []
            imgpoints = [] 
            for img in images:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                found, corners = cv.findChessboardCorners(gray, (11, 8), None)

                if found:
                    objpoints.append(objp)
                    refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                    imgpoints.append(refinedCorners)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            print("Distortion:")
            print(dist)

    def show_result(self):
        images = self.imageStorer.load_images()
        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []
        imgpoints = []
        for img in images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (11, 8), None)

            if found == True:
                objpoints.append(objp)
                refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(refinedCorners)
        
        ret, mtx, dist, rvecs, tvecs  = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        for img in images:
            undistorted_img = cv.undistort(img, mtx, dist, None, None)

            cv.namedWindow('Distored', cv.WINDOW_NORMAL)
            cv.resizeWindow('Distored', 1000, 1000)
            cv.imshow('Distored', img)
            
            cv.namedWindow('Undistorted', cv.WINDOW_NORMAL)
            cv.resizeWindow('Undistorted', 1000, 1000)
            cv.imshow('Undistorted', undistorted_img)
            cv.waitKey()
        cv.destroyAllWindows()
        
