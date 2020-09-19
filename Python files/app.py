from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from scipy.signal import convolve
# Define a flask app
app = Flask(__name__)
thre = 100
# Model saved with Keras model.save()
#MODEL_PATH = r'C:\Users\user\Desktop\model fashion mnist.h5'
model = load_model(r"C:\Users\user\Desktop\model.h5")
# Load your trained model
#model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


result_out = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
             #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
             #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
             #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
             #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                 pass
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def model_predict(img_path, model):
    #print(img_path)
    img = cv2.imread(img_path,0)
    #print(img)
    img = preprocessing(img)
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    print(type(x[0][0][0][0]))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    print(x.shape)
    x = x.astype(np.float32)
    preds = model.predict(x)
    preds = preds.argmax(axis=-1)  
    print(preds)
    return preds

def blurry(image):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(28,28))
    #print(image.shape)
    return cv2.Laplacian(image, cv2.CV_32F).var()

def preprocessing(img):
    img = img.astype(np.float32)
    t = blurry(img)
    #print(t)
    if t > thre:
        img = img.astype(np.uint8)
        edges = cv2.Canny(img,100,200)
    else:
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        #Calculate of x,y gradient
        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
    #Calculate normalized gradient magnitude and theta
        g = np.hypot(Ix, Iy)
    #G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
    #return (G, theta)
        nms = non_max_suppression(g,theta)
        #print(nms)
        edge,weak,strong = threshold(nms)
        #print(weak)
        edges = hysteresis(edge,weak=weak)
        #edges = edges > 100

    edges = edges.astype(np.float64)
    edges = cv2.resize(edges,(28,28))
    #edges = edges > 50
    return edges

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        #print(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(r"E:\\")
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        #file_path = r"E:\fashion_mnist_classification_challenge-dataset\Fashion_MNIST\test\1.jpg"
        # Make prediction
        model = load_model(r"C:\Users\user\Desktop\model.h5")

        preds = model_predict(file_path, model)
        #print(preds)
        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1]) 
#str(result_out[pred_class])              # Convert to string
        class_out = result_out[int(preds)]
        return str(class_out)
    return None


if __name__ == '__main__':
    app.run(debug=False)