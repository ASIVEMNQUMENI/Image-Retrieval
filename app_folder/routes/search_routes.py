from flask import Blueprint, request, jsonify
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from app_folder.utils_folder.similarity import get_most_similar_images, describe_top_images
from app_folder.services.speech_service import SpeechService

# Initialize the Flask Blueprint
search_bp = Blueprint('search', __name__)

# Load the pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch32"  # You can use other CLIP variants as well
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

# Initialize the SpeechService
speech_service = SpeechService()

@search_bp.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    if query:
        # Preprocess the text query and generate embeddings
        inputs = clip_processor(text=query, return_tensors="pt", padding=True)
        text_embedding = clip_model.get_text_features(**inputs).detach().numpy()
        
        # Ensure the embedding is 2D
        text_embedding = np.squeeze(text_embedding)  # Remove extra dimensions
        text_embedding = text_embedding.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Get the most similar images and their descriptions
        top_images = get_most_similar_images(text_embedding)
        descriptions = describe_top_images(top_images)  # Add descriptions
        return jsonify({"top_images": top_images, "descriptions": descriptions})
    return jsonify({"top_images": [], "descriptions": []})

@search_bp.route('/search_voice', methods=['GET'])
def search_voice():
    query = speech_service.recognize_speech()
    if query:
        # Preprocess the text query and generate embeddings
        inputs = clip_processor(text=query, return_tensors="pt", padding=True)
        text_embedding = clip_model.get_text_features(**inputs).detach().numpy()
        
        # Ensure the embedding is 2D
        text_embedding = np.squeeze(text_embedding)  # Remove extra dimensions
        text_embedding = text_embedding.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Get the most similar images and their descriptions
        top_images = get_most_similar_images(text_embedding)
        descriptions = describe_top_images(top_images)  # Add descriptions
        
        # Convert descriptions to speech
        description_text = " ".join([desc["description"] for desc in descriptions])
        speech_service.speak(description_text)
        
        return jsonify({"top_images": top_images, "descriptions": descriptions})
    speech_service.speak("Sorry, I didn't catch that.")
    return jsonify({"top_images": [], "descriptions": []})