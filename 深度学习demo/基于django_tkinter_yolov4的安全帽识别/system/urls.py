from django.contrib import admin
from django.urls import path, include
import system.views as views

urlpatterns = [
    path('', views.login, name='login'),
    path('index/', views.index, name='index'),
    path('GUI/video', views.GUI_video, name='GUI_video'),
    path('more/', views.more, name='more'),
    path('profile/', views.profile, name='profile'),
]