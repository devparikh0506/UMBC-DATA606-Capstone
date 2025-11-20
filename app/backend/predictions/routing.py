from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Support both training runs (03T) and evaluation runs (04E, 05E)
    re_path(r'^ws/predict/(?P<subject_id>B\d{2})/(?P<run_id>\d{2}[ET])/$', consumers.PredictionConsumer.as_asgi()),
]

