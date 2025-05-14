from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Preprocess image function
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0,1] range
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = preprocess_image(image)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        return jsonify({'class': int(predicted_class), 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
# codec
