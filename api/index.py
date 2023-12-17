from flask import Flask, render_template, request, jsonify
import keras.models
from keras.preprocessing.image import array_to_img
import re
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("data/model.h5")

@app.route('/')
def home():
    return render_template('uploader.html') 

@app.route('/recognize', methods=['POST'])
def recognize():
    img_size = (28, 28)
    
    data = BytesIO(request.files.get('file').read())
    image = Image.open(data).resize(img_size).convert('L')
     
    
    image_array = np.array(image, dtype='float32')
    image_array = image_array/ 255.0

    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    predict = model.predict(image_array)

    print('predict', np.argmax(predict))
    
    return jsonify(predict=str(np.argmax(predict)))