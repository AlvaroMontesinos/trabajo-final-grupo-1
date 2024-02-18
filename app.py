from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('ml/my_model')

target_labels = ["Gato", "Perro"]

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    new_file = request.files['file']
    target_path = os.path.join("upload", new_file.filename)
    new_file.save(target_path)

    image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(image, (100, 100))

    image_data = data_validation(resized_image)

    prediction = model.predict(image_data)
    predicted_label = np.argmax(prediction, axis=1)[0]

    return f"Edsf un {target_labels[predicted_label]} ;) !!"

def data_validation(image):
    image = np.expand_dims(image, axis=0)

    image = tf.cast(image, tf.float32)

    image /= 255.0

    return image

if __name__ == '__main__':
    app.run(debug=True, port=5002)
