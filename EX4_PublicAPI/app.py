from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from PIL import Image
import io
import base64
from utils import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def Predict():
    try:
        # Get base64 image
        data = request.get_json()
        base64_image = data['image']
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        pre = predict(model, image, loaded_class_vectors)

        return jsonify({'Class': pre})
    except Exception as e:
        return jsonify({'error': f'Exception: {str(e)}'}), 500

if __name__ == '__main__':
    app.run()
