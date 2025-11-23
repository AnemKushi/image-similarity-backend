import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load ResNet50 model pre-trained on ImageNet, without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_path):
    """
    Extracts features from an X-ray image using ResNet50 CNN.
    This provides more discriminative features for different body parts.
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # ResNet50 expects 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50

    # Extract features
    features = model.predict(img_array, verbose=0)
    features = features.flatten()  # Flatten to 1D vector

    # Normalize the feature vector to unit length for cosine similarity
    norm_val = np.linalg.norm(features)
    if norm_val > 0:
        features = features / norm_val

    return features
