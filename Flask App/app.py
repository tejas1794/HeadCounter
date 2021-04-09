# Imports
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import PIL.Image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np

# Load Model
model = load_model("headcount_model.h5")

# Setup Flask app with template_folder path as "pages/"
app = Flask(__name__, template_folder="pages")
app.secret_key = "8sd9fh39fgh398fh3"
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(imagepath):
    """Takes a valid 64bit image file, preprocessed and returns the final prediction"""
    img = image.load_img(imagepath, target_size=(224, 224))
    img = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    prediction = model.predict(x)
    return np.concatenate(prediction)


def allowed_file(filename):
    """Take a filename and returns whether the file is of correct format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Default route handling
@app.route('/', methods=['GET', 'POST'])
def initialize():
    """Handles home page and post calls retrieving the image query"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print('upload_image filename: ' + filename)
            flash('Image successfully uploaded - see our prediction below:')
            result = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', filename=filename, prediction=str(round(result[0]/100)))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html')


# Start app
if __name__ == '__main__':
    app.run(debug=True)
