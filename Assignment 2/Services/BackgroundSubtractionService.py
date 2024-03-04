import cv2 as cv

from Store.ImageVideoStorer import ImageVideoStorer


class BackgroundSubtractionService:
    def __init__(self, image_video_storer: ImageVideoStorer):
        self.image_video_storer = image_video_storer
        
    def background_subtraction(self):
        
        file_name= self.image_video_storer.load_video()
        subtractor = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
        cap = cv.VideoCapture(file_name)

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            blurred_frame = cv.GaussianBlur(frame, (5, 5), 0)
            mask = subtractor.apply(blurred_frame)
            foreground_objects = cv.bitwise_and(frame, frame, mask=mask)
            
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            combined_frame = cv.hconcat([frame, mask_bgr, foreground_objects])

            cv.imshow('Background Subtraction', combined_frame)
            
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv.destroyAllWindows()
