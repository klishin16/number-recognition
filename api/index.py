from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
# model = keras.models.load_model("data/model.h5")
interpreter = tflite.Interpreter(model_path="data/model.tflite")
interpreter.get_signature_list()


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

    lite_model = interpreter.get_signature_runner('serving_default')

    print(interpreter.get_signature_list())

    predict = lite_model(conv2d_3_input=image_array)['dense_1']

    print(predict)

    print('predict', np.argmax(predict))
    
    return jsonify(predict=str(np.argmax(predict)))