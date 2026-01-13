import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
from PIL import Image

from ml.model import GestureCNN
import ml.config as config

class GesturePredictor():

    def __init__(self, path_to_trained_model, mean_norm, std_norm, device, img_size, class_names):

        self.model = GestureCNN()
        self.device = device
        self.model.load_state_dict(torch.load(path_to_trained_model, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

        self.transformation = transformation
        self.img_size = img_size

        self.class_names = class_names
    
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

    def predict(self, frame, bbox):
        """
        Get gesture prediction with confidence.
        """
        with torch.no_grad():
            img_tensor = self.preprocess_image(frame, bbox)

            raw_logit_output = self.model(img_tensor)

            predicted_class_probabilities = F.softmax(raw_logit_output, dim=1)

            class_idx = torch.argmax(predicted_class_probabilities, dim=1)

            confidence = predicted_class_probabilities[0, class_idx].item()
            class_idx = class_idx.item()

            gesture_name = self.class_names[class_idx]

            return (gesture_name, confidence)
    
    def predict_with_threshold(self, frame, bbox, threshold=0.8):
        """
        Get gesture prediction only if confidence above threshold.
        """
        gesture_name, confidence = self.predict(frame, bbox)
        
        if confidence < threshold:
            return 'no_gesture'
        
        return gesture_name


        