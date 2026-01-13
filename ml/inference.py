import torch

from ml.model import GestureCNN
import ml.config as config

class GesturePredictor():

    def __init__(self, model, path_to_trained_model, device):

        self.model = model
        self.device = device
        self.model.load_state_dict(torch.load(path_to_trained_model, map_location=self.device))
    
    def preprocess_image(self, frame, bbox):
        """
        Get an OpenCV numpy frame, cropout out region of the hand, resize it and return as a Normalized Tensor
        """

        #Here, the input parameter frame will be an OpenCV (i.e. BGR) Numpy array of pixels. 
        #This needs to be processed and returned in the tensor format the model expects.

        