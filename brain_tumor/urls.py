from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name="home"),      
    path('service/', views.service, name='service'),
    path('contact/', views.contact, name='contact'),
    path('followup/', views.followup, name='followup'),

    #functions
    #for prediction
    path('scan/', views.scan_image, name='scan_image'),
    #for patient list
    path('patients/', views.patient_list, name='patient_list'),
    path('login/', views.login_view, name='login'),
    path('send-email/<int:patient_id>/', views.send_email, name='send_email'),
    path('search_patient/', views.search_patient, name='search_patient'),
    
]