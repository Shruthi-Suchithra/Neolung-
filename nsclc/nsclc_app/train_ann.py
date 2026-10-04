import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model_factory import build_ann_model
import joblib

def train_ann():
    # Paths
    base_dir = r"c:\Users\megha\Desktop\nsclc_ga_svm\dataset"
    train_path = os.path.join(base_dir, "clinical_dataset_train.csv")
    test_path = os.path.join(base_dir, "clinical dataset_test.csv")
    
    model_save_dir = r"c:\Users\megha\Desktop\nsclc_ga_svm\nsclc\media\models"
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, "nsclc_ann.h5")
    scaler_save_path = os.path.join(model_save_dir, "ann_scaler.pkl")
    encoders_save_path = os.path.join(model_save_dir, "ann_encoders.pkl")

    # Load Data
    print(f"Loading data from {train_path} and {test_path}...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Combine for consistent preprocessing (careful to split back later or just fit on train)
    # We will fit on train, transform test.
    
    # Preprocessing
    # Features: Age, Gender, Smoking_Status, EGFR_Mutation, ALK_Rearrangement, KRAS_Mutation, ROS1_Rearrangement
    # Target: NSCLC_Presence
    
    target_col = 'NSCLC_Presence'
    drop_cols = ['Patient_ID']
    
    # Prepare X, y
    X_train = df_train.drop(columns=[target_col] + drop_cols, errors='ignore')
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col] + drop_cols, errors='ignore')
    y_test = df_test[target_col]
    
    # Identify Categorical and Numerical columns
    cat_cols = ['Gender', 'Smoking_Status']
    num_cols = ['Age']
    
    # Encoders
    encoders = {}
    
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            # Fit on combined unique values to ensure all categories are covered if possible, 
            # or just handle unknown in real app. For now fit on train + test union to be safe for this specific dataset
            # In production, handle unseen labels.
            all_vals = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_vals)
            
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            encoders[col] = le
            
    # Scaler
    scaler = StandardScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save Preprocessors
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(encoders, encoders_save_path)
    print("Preprocessors saved.")
    
    # Convert to numpy
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)
    
    input_dim = X_train.shape[1]
    
    # Build Model
    # Binary classification (NSCLC Presence 0 or 1)
    # Using num_classes=1 for sigmoid output
    model = build_ann_model(input_dim=input_dim, num_classes=1)
    
    model.summary()
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=8,
        verbose=1
    )
    
    # Results
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save Model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_ann()
