from PyQt5 import QtWidgets


class ImageVideoStorer:

    def __init__(self):
        self.video_path = None
        self.image_path = None
        

    def store_video(self)->bool:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        
        if not file_name:
            return False
        
        self.video_path = file_name
        return True
    
    def load_video(self)->str:
        if self.video_path:
            return self.video_path
        return ""
    
    def store_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.png)")
        
        if not file_name:
            return False
        
        self.image_path = file_name
        return True
    
    def load_image(self)->str:
        if self.image_path:
            return self.image_path
        return ""