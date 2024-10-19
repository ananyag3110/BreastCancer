'''from __future__ import division, print_function
# coding=utf-8

import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
 

# Define a flask app
app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/Deep_Learn.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust optimizer and loss as needed
model.make_predict_function()       

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(640, 640))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image

    # Make predictions
    preds = model.predict(x)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Create uploads folder if it doesn't exist
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)

        # Threshold at 0.5 for binary classification
        pred_class = (preds > 0.36).astype(int)

        # Construct the result message
        result = f"Predicted class: {pred_class[0][0]}"  # Extract scalar value from array
        

        
        return result
    
    return None

if __name__ == '__main__':
    app.run(debug=True)
'''

from __future__ import division, print_function
# coding=utf-8
import pickle
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS

# Define a flask app
app = Flask(__name__)
CORS(app)

# Model saved 
DL_MODEL_PATH = 'models/Deep_Learn.h5'
ML_MODEL_PATH = 'models/ML.pkl'

# Load your trained model
model = load_model(DL_MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust optimizer and loss as needed
model.make_predict_function()       

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(640, 640))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image

    # Make predictions
    preds = model.predict(x)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Create uploads folder if it doesn't exist
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)

        # Threshold at 0.5 for binary classification
        pred_class = (preds > 0.36).astype(int)

        # Construct the result message
        result = f"Predicted class: {pred_class[0][0]}"  # Extract scalar value from array
        return result
    
    return None

# Define your predict function here.
def predict(data):
    # Load the pickled model.
    model = pickle.load(open(ML_MODEL_PATH, 'rb'))

    # Make a prediction on the input data.
    prediction = model.predict(data)
    return prediction

@app.route('/predictform', methods=['POST'])
def predict_endpoint():
    # Get data from the POST request.
    request_data = request.json  # Access the entire JSON object
    input_data = request_data.get('data')  # Access the 'data' key

    if input_data is None:
        return jsonify({'error': 'Data not provided in the request.'}), 400

# Call the predict function.
    result = predict(np.array([input_data]))  # Wrap the input data in a NumPy array

    # Send the result to the front end.
    return jsonify({'result': result.tolist()})  # Convert NumPy array to a list for JSON serialization


if __name__ == '__main__':
    app.run(debug=True)
