import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, send_from_directory
from app_folder.config_folder.config import Config_
from app_folder.models.clip_model import ClipModel
from app_folder.models.blip_model import CaptionService

# Initialize configuration and models
config = Config_()
clip_model = ClipModel()
caption_service = CaptionService()

# Initialize Flask app
app = Flask(__name__)

def get_most_similar_images(text_embedding, top_k=5):
    """
    Compute the most similar images to the given text embedding.

    Args:
        text_embedding (torch.Tensor or np.ndarray): Embedding of the text query.
        top_k (int): Number of top similar images to return.

    Returns:
        list: A list of dictionaries containing the top images, their captions, and similarity scores.
    """
    similarities = []

    # Ensure text_embedding is a 2D NumPy array
    if isinstance(text_embedding, torch.Tensor):
        text_embedding_np = text_embedding.detach().cpu().numpy()
    else:
        text_embedding_np = text_embedding  # Assume it's already a NumPy array

    # Flatten the text embedding to ensure it's 2D
    if text_embedding_np.ndim > 2:
        text_embedding_np = text_embedding_np.reshape(text_embedding_np.shape[0], -1)

    # Iterate through all images in the folder
    for image_file in os.listdir(config.IMAGE_FOLDER):
        # Check if the file is an image
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(config.IMAGE_FOLDER, image_file)

            # Get the image embedding
            image_embedding = clip_model.get_image_embedding(image_path)
            if image_embedding is None:
                continue

            # Convert image_embedding to NumPy array and ensure it's 2D
            if isinstance(image_embedding, torch.Tensor):
                image_embedding_np = image_embedding.detach().cpu().numpy()
            else:
                image_embedding_np = image_embedding  # Assume it's already a NumPy array

            # Flatten the image embedding to ensure it's 2D
            if image_embedding_np.ndim > 2:
                image_embedding_np = image_embedding_np.reshape(image_embedding_np.shape[0], -1)

            # Compute cosine similarity
            similarity = cosine_similarity(text_embedding_np, image_embedding_np)[0][0]
            similarities.append((image_file, similarity))

    # Sort images by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Prepare the top images with captions
    top_images = []
    for i, (image_file, similarity) in enumerate(similarities[:top_k]):
        image_path = os.path.join(config.IMAGE_FOLDER, image_file)

        # Generate a caption for the image using the BLIP model
        caption = caption_service.generate_caption(image_path)

        # Append the result to the top images list
        top_images.append({
            "label": i + 1,  # Rank of the image
            "image": image_file,  # Filename of the image
            "caption": caption,  # Generated caption
            "similarity": float(similarity)  # Similarity score
        })

    return top_images

def describe_top_images(top_images):
    """
    Describe the top images by printing or logging their details.

    Args:
        top_images (list): A list of dictionaries containing the top images, their captions, and similarity scores.
    """
    for image_info in top_images:
        print(f"Rank: {image_info['label']}")
        print(f"Image: {image_info['image']}")
        print(f"Caption: {image_info['caption']}")
        print(f"Similarity Score: {image_info['similarity']:.4f}")
        print("-" * 40)

@app.route('/')
def index():
    """
    Render the main page with the top similar images and their descriptions.
    """
    # Example text embedding (replace with your actual text embedding)
    text_embedding = clip_model.get_text_embedding("A beautiful sunset over the mountains")
    
    # Get the top similar images
    top_images = get_most_similar_images(text_embedding, top_k=5)
    
    # Log the top images to the console
    describe_top_images(top_images)
    
    # Render the results in an HTML template
    return render_template('index.html', top_images=top_images)

@app.route('/images/<filename>')
def serve_image(filename):
    """
    Serve images from the IMAGE_FOLDER directory.
    """
    return send_from_directory(config.IMAGE_FOLDER, filename)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)