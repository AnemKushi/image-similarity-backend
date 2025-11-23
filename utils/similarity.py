import numpy as np

def calculate_similarity(features1, features2):
    """
    Calculates similarity between two feature vectors using cosine similarity.
    Returns the raw cosine similarity value (-1 to 1) for debugging.
    """
    if features1 is None or features2 is None:
        return 0.0

    # Cosine similarity (features are already normalized)
    dot_product = np.dot(features1, features2)
    return round(dot_product, 4)  # Return raw cosine value
