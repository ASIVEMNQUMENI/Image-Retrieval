from flask import Blueprint, request, jsonify, current_app
import os
import torch
from app_folder.embeddings import get_text_embeddings
from app_folder.similarity import get_most_similar_images
from app_folder.speech_service import recognize_speech_from_mic, speak_text

search_bp = Blueprint('search', __name__)

@search_bp.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"top_images": []})
    text_embedding = get_text_embeddings(query)
    image_folder = os.path.join(current_app.static_folder, current_app.config['IMAGE_FOLDER'])
    top_images = get_most_similar_images(text_embedding, image_folder)
    return jsonify({"top_images": top_images})

@search_bp.route('/search_voice', methods=['GET'])
def search_voice():
    query = recognize_speech_from_mic()
    if not query:
        speak_text("Sorry, I didn't catch that.")
        return jsonify({"top_images": []})
    text_embedding = get_text_embeddings(query)
    image_folder = os.path.join(current_app.static_folder, current_app.config['IMAGE_FOLDER'])
    top_images = get_most_similar_images(text_embedding, image_folder)
    descriptions = [f"Image {img['label']}" for img in top_images]
    speak_text(" ".join(descriptions))
    return jsonify({"top_images": top_images})