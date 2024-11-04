from django.db import models

# Create your models here.
class Patient(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    email = models.EmailField(max_length=100, blank=True)
    diagnosis = models.CharField(max_length=1000, blank=True)  # e.g., Tumor/No Tumor
    

    def __str__(self):
        return self.name

class MRIImage(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='mri_images')
    mri_image = models.ImageField(upload_to='mri_images/')
    pre_image = models.ImageField(upload_to='mri_images/predicted_img/', blank=True, null=True)
    date_taken = models.DateField(auto_now_add=True)
    
class Login(models.Model):
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)

    def __str__(self):
        return self.username