import cv2

from Store.ImageStorer import ImageStorer


class SIFTService:
    def __init__(self,image_storer:ImageStorer) -> None:
        self.imageStorer = image_storer

    def showKeypoints(self):
        img1 = self.imageStorer.load_sift_image1()
        grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT.create()
        keypoints, _ = sift.detectAndCompute(grayImg1,None)
        keypoints_image = cv2.drawKeypoints(grayImg1,keypoints,None,color=(0,255,0))
        
        cv2.namedWindow('SIFT keypoints', 0)
        cv2.resizeWindow('SIFT keypoints', 500, 500)
        cv2.imshow('SIFT keypoints', keypoints_image)
        
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        
    def findMatchedKeypoints(self):
        img1 = self.imageStorer.load_sift_image1()
        img2 = self.imageStorer.load_sift_image2()

        grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT.create()
        keypoints1, descriptors1 = sift.detectAndCompute(grayImg1,None)
        keypoints2, descriptors2 = sift.detectAndCompute(grayImg2,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        img_matches = cv2.drawMatchesKnn(img1, keypoints1, 
                                         img2, keypoints2, 
                                         good, 
                                         None, 
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.namedWindow('Matched Keypoints', 0)
        cv2.resizeWindow('Matched Keypoints', 1000, 500)
        cv2.imshow('Matched Keypoints', img_matches)

        cv2.waitKey(0)
        cv2.destroyAllWindows()