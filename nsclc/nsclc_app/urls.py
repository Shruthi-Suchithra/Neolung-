from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('register/',views.register,name='register'),
    path('login/',views.signin,name='login'),
    path('logout/', views.signout, name='logout'),
    path('chatbot/', views.chat, name='chatbot'),
    path('predict/', views.predict_nsclc, name='predict_nsclc'),
    path('predict-clinical/', views.predict_clinical, name='predict_clinical'),
    path('detection-options/', views.detection_landing, name='detection_landing'),
    path('dashboard/', views.dashboard, name='dashboard'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)