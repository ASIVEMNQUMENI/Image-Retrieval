import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app_folder.embeddings import get_image_embeddings

def get_most_similar_images(text_embedding, image_folder, top_k=5):
    similarities = []
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            image_embedding = get_image_embeddings(image_path)
            if image_embedding is None:
                continue
            image_embedding = image_embedding.cpu().numpy()
            text_embedding_np = text_embedding.cpu().numpy()
            similarity = cosine_similarity(text_embedding_np, image_embedding)
            similarities.append((image_file, similarity[0][0]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_images = [{"label": i + 1, "image": img[0]} for i, img in enumerate(similarities[:top_k])]
    return top_images