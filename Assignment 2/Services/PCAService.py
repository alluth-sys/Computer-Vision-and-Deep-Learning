import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from Store.ImageVideoStorer import ImageVideoStorer


class PCAService():
    def __init__(self, image_video_storer: ImageVideoStorer):
        self.image_video_storer = image_video_storer
        
    def apply(self):
        image_path = self.image_video_storer.load_image()
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        
        if original_image is None:
            print("Error: Image not found")
            return
        else:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            normalized_image = gray_image / 255.0
            h, w = normalized_image.shape
            
            mse_threshold = 3.0
            n_components = 0
            reconstructed_image = None

            for n in range(1,min(w, h) + 1):
                pca = PCA(n_components=n)
                transformed_data = pca.fit_transform(normalized_image)
                reconstructed_data = pca.inverse_transform(transformed_data)
                
                reconstructed_rescaled = (reconstructed_data * 255).astype(np.uint)
                
                mse = mean_squared_error(gray_image, reconstructed_rescaled)
            
                if mse <= mse_threshold:
                    reconstructed_image = reconstructed_data
                    n_components = n
                    break
            
            
            print(f"n= {n_components}")

            cv2.imshow("Grayscale Image", gray_image)
            cv2.imshow(f"Reconstructed Image", reconstructed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()