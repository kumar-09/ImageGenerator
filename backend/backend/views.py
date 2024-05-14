from rest_framework.decorators import api_view
import requests
from django.http import HttpResponse




@api_view(['GET'])
def refresh():
    ai_server_url = 'http://localhost:5001/generate_image'
    response = requests.get(ai_server_url)
    return HttpResponse(response.content, content_type='image/png')