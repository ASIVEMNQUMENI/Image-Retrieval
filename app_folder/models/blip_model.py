from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from app_folder.config_folder.config import Config_

class CaptionService:
    def __init__(self):
        self.config = Config_()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.config.DEVICE)

    def generate_caption(self, image_path):
        try:
            # Load the image
            image = Image.open(image_path).convert("RGB")
            # Preprocess the image
            inputs = self.processor(image, return_tensors="pt").to(self.config.DEVICE)
            # Generate caption
            with torch.no_grad():
                caption_ids = self.model.generate(**inputs)
            caption = self.processor.decode(caption_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None