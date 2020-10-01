from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import json

with open('./model/label.json') as f:
  label = json.load(f)

model = load_model('./model/model_InceptionV3.h5')

# Create your views here.
def home(request):
    return render(request, "classifier_app/home.html")
def classifyImage(request):
    if request.method == 'POST':
        image = request.FILES['image']
        fs = FileSystemStorage()
        imageName = fs.save(image.name, image)
        imageName = fs.url(imageName)
        loc = '.' + imageName
        
        img = load_img(loc, target_size=(100, 100))
        img_arry = img_to_array(img)
        to_pred = np.expand_dims(img_arry, axis = 0)
        prep = preprocess_input(to_pred)
        prediction = model.predict(prep)
        percentage = np.max(prediction)
        prediction = np.argmax(prediction)

        if(percentage > 0.5):
            ans = label['0'][str(prediction)].split()[0]
        else:
            ans = "Sorry, unable to classify"

        context = {
            'imageName': imageName,
            'label': ans
        }
        return render(request, "classifier_app/home.html", context)

