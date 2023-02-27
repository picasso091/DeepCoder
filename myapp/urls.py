from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views
urlpatterns = [
    path('', views.home, name = 'home'),
    path('home/', views.home, name = 'home'),
    path('result/',views.result, name = 'result' ),
    path('api_expose/',views.api_expose, name = 'api_expose' )
]
urlpatterns += staticfiles_urlpatterns()