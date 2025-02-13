from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from app_folder.config_folder.config import Config_
import torch

class ClipModel:
    def __init__(self):
        self.config = Config_()
        self.device = self.config.DEVICE
        
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

    def get_image_embedding(self, image_path):
        """
        Get the embedding for an image using CLIP.
        """
        try:
            from PIL import Image
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        """
        Get the embedding for a text query using CLIP.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)