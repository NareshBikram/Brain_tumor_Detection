from django.contrib import admin

# Register your models here.
from .models import Patient,MRIImage

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'age', 'diagnosis')  # Removed 'prediction_date' if it's not in Patient model
    search_fields = ('name', 'diagnosis')
    list_filter = ('diagnosis',)

class MRIImageInline(admin.TabularInline):
    model = MRIImage
    extra = 1  # Number of empty forms to display

# Add MRIImageInline to PatientAdmin to manage MRI images within Patient
@admin.register(MRIImage)
class MRIImageAdmin(admin.ModelAdmin):
    list_display = ('patient', 'date_taken', 'mri_image', 'pre_image')
    list_filter = ('date_taken', 'patient')
    search_fields = ('patient__name',)  # Allows searching by patient name

# Attach MRIImageInline to PatientAdmin
PatientAdmin.inlines = [MRIImageInline]

from .models import Login

@admin.register(Login)
class LoginAdmin(admin.ModelAdmin):
    list_display = ('username',) 