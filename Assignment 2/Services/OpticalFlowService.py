import cv2
import numpy as np

from Store.ImageVideoStorer import ImageVideoStorer


class OpticalFlowService:
    def __init__(self, image_video_storer:ImageVideoStorer):
        self.image_video_storer = image_video_storer

    def find_good_features_to_track(self):
        video_path = self.image_video_storer.load_video()
        cap = cv2.VideoCapture(video_path)
        
        ret, frame = cap.read()
        if not ret:
            return
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        max_corners = 1
        quality_level = 0.3
        min_distance = 7
        block_size = 7
        points = cv2.goodFeaturesToTrack(gray_frame, max_corners, quality_level, min_distance, blockSize=block_size)

        if points is not None:
            frame_center_bottom = (frame.shape[1] / 2, frame.shape[0])
            nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(frame_center_bottom) - p[0]))

            cross_length = 20
            cross_thickness = 4
            x, y = nearest_point.ravel()
            cv2.line(frame, (int(x - cross_length / 2), int(y)), (int(x + cross_length / 2), int(y)), (0, 0, 255), cross_thickness)
            cv2.line(frame, (int(x), int(y - cross_length / 2)), (int(x), int(y + cross_length / 2)), (0, 0, 255), cross_thickness)

        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Good Features to Track", resized_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cap.release()
        
    def draw_trajectory(self):
        # Load the video
        cap = cv2.VideoCapture(self.image_video_storer.load_video())

        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            return

        # Convert the frame to grayscale for feature detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Parameters for Shi-Tomasi corner detection (goodFeaturesToTrack)
        max_corners = 1
        quality_level = 0.3
        min_distance = 7
        block_size = 7

        points = cv2.goodFeaturesToTrack(gray_frame, max_corners, quality_level, min_distance, blockSize=block_size)
        
        if points is None:
            return

        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)
        
        cross_length = 20
        cross_thickness = 4
        scale_percent = 50

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        while True:
            ret, new_frame = cap.read()
            if not ret:
                break

            new_gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

            # Calculate the optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(gray_frame, new_gray_frame, points, None, **lk_params)

            # Select good points
            good_new = new_points[status == 1]
            good_old = points[status == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x1, y1 = new.ravel()
                x2, y2 = old.ravel()
                mask = cv2.line(mask, (x1, y1), (x2, y2), (0, 100, 255), 2)
                cv2.line(new_frame, (int(x1 - cross_length / 2), int(y1)), (int(x1 + cross_length / 2), int(y1)), (0, 0, 255), cross_thickness)
                cv2.line(new_frame, (int(x1), int(y1 - cross_length / 2)), (int(x1), int(y1 + cross_length / 2)), (0, 0, 255), cross_thickness)


            # Overlay the mask on the original frame
            output = cv2.add(new_frame, mask)
            
            # Resize the output
            new_width = int(output.shape[1] * scale_percent / 100)
            new_height = int(output.shape[0] * scale_percent / 100)
            resized_output = cv2.resize(output, (new_width, new_height))

            # Update the previous frame and previous points
            gray_frame = new_gray_frame.copy()
            points = good_new.reshape(-1, 1, 2)

            # Display the output
            cv2.imshow("Trajectory", resized_output)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        