from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

app = Flask(__name__)
model = load_model("steel_defect_model.h5")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Your class labels in order (match training order)
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            image = load_img(file_path, target_size=(200, 200))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            pred = model.predict(image)
            predicted_class = CLASS_NAMES[np.argmax(pred)]

            prediction = predicted_class
            image_path = file_path

    return render_template("index.html", prediction=prediction, image_path=image_path)

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
