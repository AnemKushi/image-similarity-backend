from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import os
import pickle
import uuid
import logging
from utils.feature_extraction import extract_features
from utils.similarity import calculate_similarity
from models import db, User
from auth import register, login

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this to a secure key
db.init_app(app)
jwt = JWTManager(app)

# Path to dataset folder
DATASET_PATH = "dataset"

# Load precomputed features and classifier
with open("features.pkl", "rb") as f:
    features_db = pickle.load(f)
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return "🩻 X-Ray Similarity Backend Running Successfully!"

@app.route('/welcome', methods=['GET'])
def welcome():
    logger.info(f"Request received: {request.method} {request.path}")
    return jsonify({"message": "Welcome to the Image Similarity API!"})

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        if 'image' not in request.files:
            logger.error("No image uploaded in request")
            return jsonify({"error": "No image uploaded"}), 400

        uploaded_image = request.files['image']

        if not uploaded_image.filename:
            logger.error("Invalid image filename")
            return jsonify({"error": "Invalid image filename"}), 400

        image_path = os.path.join(app.root_path, 'static', uploaded_image.filename)
        try:
            uploaded_image.save(image_path)
            logger.info(f"Image saved to: {image_path}")
        except Exception as e:
            logger.error(f"Failed to save image {uploaded_image.filename}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

        try:
            input_features = extract_features(image_path)
            if input_features is None:
                logger.error(f"Feature extraction returned None for {image_path}")
                return jsonify({"error": "Failed to extract features from image"}), 500
            logger.info("Features extracted successfully")
        except Exception as e:
            logger.error(f"Feature extraction error for {image_path}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error detecting image: {str(e)}"}), 500

        try:
            prediction = classifier.predict([input_features])[0]
            probabilities = classifier.predict_proba([input_features])[0]
            max_prob = max(probabilities)
        except Exception as e:
            logger.error(f"Classifier prediction error: {str(e)}", exc_info=True)
            return jsonify({"error": "Failed to classify uploaded image"}), 500

        if max_prob < 0.8:
            logger.warning("Uploaded image does not appear to be a valid X-ray with sufficient confidence")
            return jsonify({"error": "Uploaded image does not appear to be an X-ray or the confidence is too low. Please upload a valid and clearer X-ray image."}), 400

        category = "fractured" if prediction == 1 else "normal"
        logger.info(f"Classified as: {category} with confidence {max_prob:.2f}")

        similarities = []

        for filename, dataset_features in features_db[category].items():
            if dataset_features is None or input_features is None:
                logger.error(f"None feature vector for comparison: {filename}")
                continue
            try:
                sim_score = calculate_similarity(input_features, dataset_features)
                similarities.append({"filename": filename, "similarity": sim_score})
                logger.debug(f"Similarity {sim_score} computed for {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True)

        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

        for sim in similarities:
            sim['similarity'] = float(sim['similarity'])

        # Cleanup uploaded image file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Deleted uploaded image: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded image {image_path}: {str(e)}")

        return jsonify({
            "uploaded_image": uploaded_image.filename,
            "classified_as": category,
            "results": similarities[:5]
        })
    except Exception as e:
        logger.exception(f"Unexpected error in /compare endpoint: {e}")
        return jsonify({"error": "Internal server error occurred"}), 500

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    if 'file1' not in request.files or 'file2' not in request.files:
        logger.error("Two images required but not provided in request")
        return jsonify({"error": "Two images required"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    unique_id1 = str(uuid.uuid4())
    unique_id2 = str(uuid.uuid4())
    ext1 = os.path.splitext(file1.filename)[1]
    ext2 = os.path.splitext(file2.filename)[1]
    unique_filename1 = f"{unique_id1}{ext1}"
    unique_filename2 = f"{unique_id2}{ext2}"

    path1 = os.path.join(app.root_path, 'static', unique_filename1)
    path2 = os.path.join(app.root_path, 'static', unique_filename2)
    try:
        file1.save(path1)
        file2.save(path2)
        logger.info(f"Saved files {unique_filename1}, {unique_filename2}")
    except Exception as e:
        logger.error(f"Failed to save uploaded files: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to save uploaded images: {str(e)}"}), 500

    try:
        features1 = extract_features(path1)
        features2 = extract_features(path2)
        if features1 is None or features2 is None:
            logger.error("Feature extraction returned None for one or both images")
            return jsonify({"error": "Failed to extract features from one or both images"}), 500
        score = calculate_similarity(features1, features2)
        percentage_score = round((score + 1) * 50, 2)
        logger.info(f"Similarity score computed: {percentage_score}")
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(path1):
                os.remove(path1)
            if os.path.exists(path2):
                os.remove(path2)
            logger.info("Temporary image files deleted")
        except Exception as e:
            logger.warning(f"Failed to delete temporary image files: {str(e)}")

    return jsonify({"similarity_score": float(percentage_score)})

@app.route('/dataset/<category>/<filename>')
def get_dataset_image(category, filename):
    return send_from_directory(os.path.join(DATASET_PATH, category), filename)

@app.route('/register', methods=['POST'])
def register_user():
    return register()

@app.route('/login', methods=['POST'])
def login_user():
    return login()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
