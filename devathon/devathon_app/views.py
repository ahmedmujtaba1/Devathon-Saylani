from django.shortcuts import render
from django.http import HttpResponse
from keras.models import load_model  
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from django.http import JsonResponse

# Create your views here.
def home(request):
    return render(request, 'index.html')

@csrf_exempt  
def get_data(request):
    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        np.set_printoptions(suppress=True)
        model = load_model("chest_infection_model.h5", compile=False)
        prediction = model.predict(uploaded_image)

        if uploaded_image:
            accuracy = 0.90
            category = "Some Category"

            response_data = {
                'accuracy': prediction,
                'category': category,
            }

            return JsonResponse(response_data)
    return JsonResponse({'error': 'Invalid request'}, status=400)
