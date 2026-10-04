from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate, login, logout
from .models import *
from django.http import JsonResponse
import chatbot
from .model_factory import build_model, predict_image, load_trained_model
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Global model variable
MODELS_DIR = os.path.join(settings.MEDIA_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'nsclc_cnn.h5')
TRAINED_MODEL = None

def load_model_globally():
    global TRAINED_MODEL
    if TRAINED_MODEL is None and os.path.exists(MODEL_PATH):
        TRAINED_MODEL = load_trained_model(MODEL_PATH)
    return TRAINED_MODEL

def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request,'home.html')

def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'dashboard.html')

def about(request):
    return render(request,'about.html')

def register(request):
    if request.method == "POST":
        profile = request.FILES.get('profile')
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        password = request.POST.get('password')
        address = request.POST.get('address')
        
        # Check if user already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'registration.html', {'error': 'Email already registered'})
            
        # Use create_user to handle password hashing and normalization
        try:
            user = User.objects.create_user(email=email, password=password, name=name, phone=phone, address=address, profile=profile)
            return redirect('login')
        except Exception as e:
            return render(request, 'registration.html', {'error': str(e)})
            
    return render(request, 'registration.html')

def signin(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Authenticate using email as username
        user = authenticate(request, username=email, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'login.html', {'error': 'Invalid email or password'})
            
    return render(request, 'login.html')

def signout(request):
    logout(request)
    return redirect('login')

def chat(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        bot_reply = chatbot.get_response(user_message)
        return JsonResponse({"response": bot_reply})
    return render(request, 'chatbot.html')

def predict_nsclc(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        
        try:
            model = load_model_globally()
            if model is None:
                return JsonResponse({'error': 'Model not trained or not found. Please train the model first.'})
                
            # Class names must match the training generator order (alphabetical)
            class_names = ['Benign cases', 'Malignant cases', 'Normal cases'] 
            
            label, confidence = predict_image(model, file_path, class_names)
            
            return render(request, 'result.html', {
                'label': label,
                'confidence': f"{confidence*100:.2f}%",
                'image_url': fs.url(filename)
            })
            
        except Exception as e:
             return JsonResponse({'error': str(e)})
             
    return render(request, 'predict.html')

def detection_landing(request):
    return render(request, 'detection_landing.html')

def load_ann_artifacts():
    import joblib
    try:
        model_path = os.path.join(MODELS_DIR, 'nsclc_ann.h5')
        scaler_path = os.path.join(MODELS_DIR, 'ann_scaler.pkl')
        encoders_path = os.path.join(MODELS_DIR, 'ann_encoders.pkl')
        
        if not os.path.exists(model_path): return None, None, None
        
        model = load_trained_model(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)
        return model, scaler, encoders
    except Exception as e:
        print(f"Error loading ANN artifacts: {e}")
        return None, None, None

def predict_clinical(request):
    if request.method == 'POST':
        try:
            # Load artifacts
            model, scaler, encoders = load_ann_artifacts()
            if not model:
                return render(request, 'predict_clinical.html', {'error': 'Clinical model not trained yet.'})
                
            # Extract data
            age = float(request.POST.get('age'))
            gender = request.POST.get('gender')
            smoking = request.POST.get('smoking_status')
            
            # Mutations (Checkbox = 'on' if checked, else None)
            egfr = 1 if request.POST.get('egfr') else 0
            alk = 1 if request.POST.get('alk') else 0
            kras = 1 if request.POST.get('kras') else 0
            ros1 = 1 if request.POST.get('ros1') else 0
            
            import pandas as pd
            import numpy as np
            
            # Create DataFrame to match training structure for encoding/scaling
            # Columns: Age, Gender, Smoking_Status, EGFR_Mutation, ALK_Rearrangement, KRAS_Mutation, ROS1_Rearrangement
            
            # Encode Categoricals
            # Note: In production we should handle unseen labels carefully.
            # Here we assume the form options match training data (Male/Female, Never/Former/Current)
            
            gender_enc = encoders['Gender'].transform([gender])[0]
            smoking_enc = encoders['Smoking_Status'].transform([smoking])[0]
            
            # Prepare Input Array (Order matters!)
            # Age, Gender-Enc, Smoking-Enc, EGFR, ALK, KRAS, ROS1
            # Wait, scaler was fitted on 'Age'.
            # Train script: Cat cols label encoded in place. Then scaler fitted on num_cols.
            
            # Let's construct the raw row and apply transforms exactly as in training
            data = {
                'Age': [age],
                'Gender': [gender],
                'Smoking_Status': [smoking],
                'EGFR_Mutation': [egfr],
                'ALK_Rearrangement': [alk],
                'KRAS_Mutation': [kras],
                'ROS1_Rearrangement': [ros1]
            }
            
            df = pd.DataFrame(data)
            
            # Encode
            df['Gender'] = encoders['Gender'].transform(df['Gender'])
            df['Smoking_Status'] = encoders['Smoking_Status'].transform(df['Smoking_Status'])
            
            # Scale Age
            df[['Age']] = scaler.transform(df[['Age']])
            
            # Convert to numpy
            X = df.values.astype(np.float32)
            
            # Predict
            pred = model.predict(X)
            confidence = float(pred[0][0])
            
            # Binary classification: 0 = No NSCLC (Benign/Normal?), 1 = NSCLC Presence
            # Train target: 'NSCLC_Presence'. 1 means present.
            
            label = "High Risk of NSCLC" if confidence > 0.5 else "Low Risk of NSCLC"
            conf_str = f"{confidence*100:.2f}%"
            
            return render(request, 'result_clinical.html', {
                'label': label,
                'confidence': conf_str,
                'input_data': data
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'predict_clinical.html', {'error': f"Error during prediction: {str(e)}"})
            
    return render(request, 'predict_clinical.html')