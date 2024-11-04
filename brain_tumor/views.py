from django.shortcuts import render, redirect, get_object_or_404

#custom imports
from django.db.models import Q

from django.contrib import messages
from .models import Login
from .forms import LoginForm
from .models import Patient, MRIImage

#for sending gmail
from django.conf import settings
from django.core.mail import EmailMessage
from django.core.files.base import ContentFile

# Create your views here.
def home(request):
    return render(request,'home.html',{})


def service(request):
    return render(request,'brain_tumor/service.html',{})

def contact(request):
    return render(request,'brain_tumor/contact.html',{})

def followup(request):
    return render(request,'brain_tumor/followup.html')


def patient_list(request):
    query = request.GET.get('query', '')
    patients = Patient.objects.filter(name__icontains=query) if query else Patient.objects.all()
    return render(request, 'brain_tumor/patient_list.html', {'patients': patients, 'query': query})


#for login form
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            # Verify the username and password
            try:
                user = Login.objects.get(username=username, password=password)
                return redirect('patient_list')  # Redirect to patient_list view on successful login
            except Login.DoesNotExist:
                messages.error(request, "Invalid username or password.")
                return redirect('login')

    else:
        form = LoginForm()

    return render(request, 'brain_tumor/login.html', {'form': form})



#for prediction of image
# views.py
from django.shortcuts import render
from django.core.files.base import ContentFile
from .models import Patient
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

def Dice_Coefficient(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred, axis=[1, 2, 3])
    union = np.sum(y_true, axis=[1, 2, 3]) + np.sum(y_pred, axis=[1, 2, 3])
    dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice

def Dice_Loss(y_true, y_pred):
    return 1.0 - Dice_Coefficient(y_true, y_pred)

# Load the UNet model once at the start
Unet_Model = load_model('brain_tumor/ml_model/segmetation_Model.h5', custom_objects={'Dice_Loss': Dice_Loss, 'Dice_Coefficient': Dice_Coefficient})

def combine_pred_original(path, dir_path=False, plot=False):
    # Load and preprocess the image
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_1 = image_gray / 255.0
    image_1 = image_1.astype(np.float32)
    image_1 = image_1.reshape(-1, 256, 256, 1)

    # Get the prediction from the model
    y_hat = Unet_Model.predict(image_1)
    y_hat = y_hat.reshape(256, 256)

    # Threshold the prediction to create a binary mask
    threshold = 0.5
    binary_mask = (y_hat > threshold).astype(int)

    # Calculate confidence values
    tumor_confidences = y_hat[binary_mask == 1]  # Extract confidence for tumor pixels
    if len(tumor_confidences) > 0:
        avg_confidence = np.mean(tumor_confidences)
        max_confidence = np.max(tumor_confidences)
        min_confidence = np.min(tumor_confidences)
        tumor_detected = True
    else:
        avg_confidence = max_confidence = min_confidence = 0
        tumor_detected = False

    # Apply the mask to the original image
    mask = (y_hat > threshold).astype(np.uint8) * 255
    final_img = np.copy(image)
    index_mask = np.where(mask == 255)
    final_img[index_mask] = [0, 0, 255]

    # Convert `final_img` to uint8 for PIL compatibility and resize
    final_img = cv2.resize(final_img, (512, 512))
    final_img = Image.fromarray(final_img.astype(np.uint8))

    if plot:
        final_img.show()

    # Return final image, detection status, and confidence values
    return final_img, tumor_detected, avg_confidence, max_confidence, min_confidence


def scan_image(request):
    if request.method == 'POST':
        # Collect form data
        name = request.POST.get('name')
        age = request.POST.get('age')
        email = request.POST.get('email')
        mri_image = request.FILES.get('image')

        # Check if patient already exists, else create new
        patient, created = Patient.objects.get_or_create(
            name=name, age=age, email=email,
            defaults={'diagnosis': ''}
        )

        # Save MRI image entry in MRIImage model
        mri_image_entry = MRIImage(patient=patient, mri_image=mri_image)
        mri_image_entry.save()

        # Perform prediction on the MRI image
        mri_image_path = mri_image_entry.mri_image.path
        result_image, tumor_detected, avg_confidence, max_confidence, min_confidence = combine_pred_original(
            path=mri_image_path, dir_path=False, plot=False
        )

        # Convert the result image to a format suitable for saving
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        image_content = ContentFile(buffer.getvalue(), name="predicted_image.png")

        # Save the predicted image in the MRIImage model
        mri_image_entry.pre_image.save("predicted_image.png", image_content)

        # Update diagnosis based on tumor detection
        if tumor_detected:
            patient_summary = (
                f"This {patient.age}-year-old patient has been diagnosed with a brain tumor. "
                "Following initial symptoms, an MRI scan confirmed the presence of a mass, "
                "located in a critical area. The scan analysis indicates a high probability of malignancy, "
                f"with an average confidence of {avg_confidence:.2f} in the detected tumor regions. "
                "The recommended course of action involves further diagnostic tests and consultation with "
                "a specialized oncology team to determine a personalized treatment plan. "
                "The treatment options may include surgery, radiation therapy, or chemotherapy, depending on "
                "the tumor’s characteristics and location."
            )
            patient.diagnosis = patient_summary
        else:
            patient_summary_no_tumor = (
                f"This {patient.age}-year-old patient has undergone an MRI scan, "
                "and the results indicate no presence of a brain tumor. "
                "While initial symptoms raised concerns, the scan has provided reassurance "
                "by confirming that there are no abnormal masses in the brain. "
                "It is essential to continue monitoring any symptoms and maintain regular follow-ups "
                "with healthcare professionals to ensure ongoing health and well-being. "
                "Preventive measures and a healthy lifestyle are recommended to support brain health."
            )
            patient.diagnosis = patient_summary_no_tumor

        patient.save()

        # Redirect to a result page with patient data and confidence scores
        return render(request, 'brain_tumor/result.html', {
            'patient': patient,
            'tumor_detected': tumor_detected,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence
        })

    return render(request, 'brain_tumor/service.html')


#sending email
def send_email(request, patient_id):
    # Get the patient data
    patient = get_object_or_404(Patient, id=patient_id)

    # Construct the email content
    subject = f"Patient Record for {patient.name}"
    message = (
        f"ID: {patient.id}\n"
        f"Name: {patient.name}\n"
        f"Age: {patient.age}\n"
        f"Email: {patient.email}\n"
        f"Diagnosis: {patient.diagnosis}\n"
    )
    
    # Create an email message
    email = EmailMessage(
        subject,
        message,
        settings.DEFAULT_FROM_EMAIL,
        [patient.email]
    )
    
    # Attach all MRI images and predicted images if available
    mri_images = patient.mri_images.all()
    for mri in mri_images:
        if mri.mri_image:
            email.attach_file(mri.mri_image.path)
        if mri.pre_image:
            email.attach_file(mri.pre_image.path)

    # Send the email
    email.send(fail_silently=False)

    # Add a success message
    messages.success(request, "Email sent successfully to the patient.")
    return redirect('patient_list')



#=======================================================================================
def search_patient(request):
    if request.method == 'POST':
        # Collect form data
        patient_id = request.POST.get('patient_id')
        mri_image = request.FILES.get('image')

        # Check if the patient ID and MRI image are provided
        if patient_id and mri_image:
            # Retrieve the patient by ID, or return 404 if not found
            patient = get_object_or_404(Patient, id=patient_id)

            # Save initial MRI image in MRIImage model
            mri_image_entry = MRIImage(patient=patient, mri_image=mri_image)
            mri_image_entry.save()

            # Perform prediction on the MRI image
            mri_image_path = mri_image_entry.mri_image.path
            result_image, tumor_detected, avg_confidence, max_confidence, min_confidence = combine_pred_original(
                path=mri_image_path, dir_path=False, plot=False
            )

            # Convert the result image to a format suitable for saving
            buffer = BytesIO()
            result_image.save(buffer, format="PNG")
            image_content = ContentFile(buffer.getvalue(), name="predicted_image.png")

            # Save the predicted image in the MRIImage model
            mri_image_entry.pre_image.save("predicted_image.png", image_content)

            # Success message for follow-up details added
            messages.success(request, "Follow-up details have been successfully added.")

            # Render a template to display patient details and prediction results
            return render(request, 'brain_tumor/result.html', {
                'patient': patient,
                'tumor_detected': tumor_detected,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
            })

    # If request method is not POST, redirect to the main service page or show an error
    return render(request, 'brain_tumor/service.html', {'error': 'Please enter a valid Patient ID and upload an image.'})