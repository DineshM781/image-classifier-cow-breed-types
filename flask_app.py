from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = 'cattle_breed_model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("ERROR: Model not found. Please check file name.")

DATASET_PATH = 'dataset/specialized'
if os.path.exists(DATASET_PATH):
    CLASS_NAMES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
else:
   
    CLASS_NAMES = ['Gir', 'Holstein', 'Jersey', 'Murrah_Buffalo', 'Sahiwal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
          image = Image.open(file.stream)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

   
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class_index = np.argmax(predictions)
        confidence = 100 * np.max(score)

        if predicted_class_index < len(CLASS_NAMES):
            result = CLASS_NAMES[predicted_class_index]
        else:
            result = "Unknown"

        return jsonify({
            'class': result, 
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
