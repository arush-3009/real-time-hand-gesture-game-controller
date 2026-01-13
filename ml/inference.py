import torch
from torchvision import transforms
import cv2
from PIL import Image

from ml.model import GestureCNN
import ml.config as config

class GesturePredictor():

    def __init__(self, model, path_to_trained_model, mean_norm, std_norm, device, img_size):

        self.model = model
        self.device = device
        self.model.load_state_dict(torch.load(path_to_trained_model, map_location=self.device))

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

        self.transformation = transformation
        self.img_size = img_size
    
    def preprocess_image(self, frame, bbox):
        """
        Get an OpenCV numpy frame, cropout out region of the hand, resize it and return as a Normalized Tensor
        """

        #Here, the input parameter frame will be an OpenCV (i.e. BGR) Numpy array of pixels. 
        #This needs to be processed and returned in the tensor format the model expects.

        x_min, y_min, x_max, y_max = bbox
        frame_cropped = frame[y_min:y_max, x_min:x_max]
        frame_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_cropped, (self.img_size, self.img_size))

        pil_image = Image.fromarray(frame_resized)

        tensor = self.transformation(pil_image)
        tensor = tensor.unsqueeze(0) #As the model expects the batch dimension too
        tensor = tensor.to(self.device)
        return tensor

    def softmax(self, logit_tensor):
        """
        Convert a tensor of logits (raw) to probabilities.
        """
        exp_tensor = torch.exp(logit_tensor)
        total_exp = exp_tensor.sum()
        return exp_tensor/total_exp

    def predict(self, frame, bbox):
        """
        Get gesture prediction with confidence.
        """
        img_tensor = self.preprocess_image(frame, bbox)

        raw_logit_output = self.model(img_tensor)

        predicted_class_probabilities = self.softmax(raw_logit_output)

        class_idx = torch.argmax(predicted_class_probabilities, dim=1)

        confidence = predicted_class_probabilities[0, class_idx].item()
        class_idx = class_idx.item()
        return (class_idx, confidence)


        