from django.shortcuts import render
from django.http import HttpResponse
from keras.models import load_model  
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from PIL import Image, ImageOps
from django.http import JsonResponse

# Create your views here.
def home(request):
    return render(request, 'index.html')

@csrf_exempt  
def get_data(request):
    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        np.set_printoptions(suppress=True)

        if uploaded_image:
            model = load_model("B:/Ahmed'sCode/Devathon-Saylani/Devathon-Saylani/keras_model.h5", compile=False)
            class_names = open("B:/Ahmed'sCode/Devathon-Saylani/Devathon-Saylani/labels.txt", "r").readlines()
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(uploaded_image).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print("Class:", class_name[2:], end="")
            response_data = {
                'accuracy': confidence_score,
                'category': class_name,
            }

            return JsonResponse(response_data)
    return JsonResponse({'error': 'Invalid request'}, status=400)
