from PIL import Image
import torch
from app_folder.clip_model import model, processor

def get_image_embeddings(image_path):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_text_embeddings(text_query):
    inputs = processor(text=text_query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)