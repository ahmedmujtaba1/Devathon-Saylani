from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    return render(request, 'index.html')

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  
def get_data(request):
    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            accuracy = 0.90
            category = "Some Category"

            response_data = {
                'accuracy': accuracy,
                'category': category,
            }

            return JsonResponse(response_data)
    return JsonResponse({'error': 'Invalid request'}, status=400)
