from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# # Load the trained model
# model_version = 1
# model_path = f"./model/{model_version}"
# loaded_model = load_model(model_path)

# Define a route to handle image classification requests
@app.route("/predict", methods=["POST"])
def predict():
    class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    # Check if the request contains an image file
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the image file from the request
    file = request.files["image"]
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No image provided"}), 400

    # Preprocess the image
    img_size = 256
    img = image.load_img(io.BytesIO(file.read()), target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    # predictions = loaded_model.predict(img_array)
    # predicted_class = np.argmax(predictions)
    # confidence=round(100*(np.max(predictions[0])),2)
    
    # Return the predicted class
    return jsonify({"class":img_size, "label": 'class_names[int(predicted_class)]', "confidence": 'confidence'})

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=4000)
