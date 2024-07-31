from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from utils import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def Predict():
    data = request.json

    base64_image = data['image']
    # Decode base64 image
    image_data = base64.b64decode(str(base64_image))
    byte = io.BytesIO(image_data)
    image = plt.imread(byte)
    
    # Preprocess image
    pre = int(predict(model, image, loaded_class_vectors))
    return jsonify({'Class': pre})

if __name__ == '__main__':
    app.run()
