from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# # Load the trained model
model_version = 1
model_path = f"./model/{model_version}"
loaded_model = load_model(model_path)

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
    
    # Load the image using Pillow
    img = Image.open(io.BytesIO(file.read()))

    # Resize the image
    img = img.resize((img_size, img_size))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Add an extra dimension to match the expected input shape (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence=round(100*(np.max(predictions[0])),2)
    
    # Return the predicted class
    return jsonify({"class":int(predicted_class), "label": class_names[int(predicted_class)], "confidence": confidence})

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=4000)
