from flask import Flask, request,render_template, json
import pickle
import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('ml/my_model')

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    new_file = request.files['file']
    target_path = os.path.join("upload",new_file.filename)
    new_file.save(target_path)

    image = cv2.imread(target_path, 0)
    image = data_validation(image)

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, 1)
    return f"Es un {predicted_label[0]} ;) !!"

def data_validation(image):
    # image = image.flatten()
    image = np.expand_dims(image, axis=0)
    # Convertir la imagen a float32 para usar valores decimales en el tensor
    image = tf.cast(image, tf.float32)
    # Dividir el tensor entre el nivel de intensidad mas alto en la imagen
    return image

if __name__ == '__main__':
     app.run(debug=True, port=5002)