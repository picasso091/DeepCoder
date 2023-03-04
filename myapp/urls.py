from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views
app_name = "app_name"
urlpatterns = [
    path(r'', views.home, name = 'home'),
    path(r'home/', views.home, name = 'home'),
    path(r'result_t5/',views.result, name = 'result' ),
    path(r'result_vanilla/',views.result, name = 'result' ),
    path(r'api_expose/',views.api_expose, name = 'api_expose' )
]
urlpatterns += staticfiles_urlpatterns()