import os

import cv2 as cv
import numpy as np

from Store.ImageStorer import ImageStorer


class AugmentedRealityService:
    def __init__(self,image_storer:ImageStorer) -> None:
        self.imageStorer = image_storer
        
    def showWordsOnBoard(self, word:str, mode:str):
        images = self.imageStorer.load_images()

        if mode == "ONBOARD":
            fs = cv.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt',cv.FILE_STORAGE_READ)
        elif mode == "VERTICAL":
             fs = cv.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_vertical.txt',cv.FILE_STORAGE_READ)

        objp = np.zeros((11*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        characterOffsets = [np.array([8,5,0],dtype=np.float32),
                            np.array([5,5,0],dtype=np.float32),
                            np.array([2,5,0],dtype=np.float32),
                            np.array([8,2,0],dtype=np.float32),
                            np.array([5,2,0],dtype=np.float32),
                            np.array([2,2,0],dtype=np.float32)
                            ]

        for img in images:
            objpoints = []
            imgpoints = []
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(gray, (11, 8), None)

            if found == True:
                objpoints.append(objp)
                refinedCorners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(refinedCorners)

                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                

                sanitizedWord = word.upper()
                wordLines = []
                for char in sanitizedWord:
                    wordLines.append(fs.getNode(char).mat())
                
                projectedWordLines = []
                for i ,wordLine in enumerate(wordLines):
                    for line in wordLine:
                        lineProjectedPoints, jac = cv.projectPoints(
                            np.add(np.array(line,dtype=np.float32),characterOffsets[i]), 
                            rvecs[0], tvecs[0], mtx, dist)
                        projectedWordLines.append(lineProjectedPoints)

                for pts in projectedWordLines:
                    imgpts = np.int32(pts).reshape(-1,2)
                    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 10)
                
                cv.namedWindow('Projected Points', 0)
                cv.resizeWindow('Projected Points', 1000, 1000)
                cv.imshow('Projected Points',img)
                cv.waitKey(1000)
        cv.destroyAllWindows()