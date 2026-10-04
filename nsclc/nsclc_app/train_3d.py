import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_generators import NiftiDataGenerator
from model_factory import build_3d_model

def train_3d_model():
    # Configuration
    DATASET_DIR = r"c:\Users\megha\Desktop\nsclc_ga_svm\dataset\cancerous"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
    MODEL_SAVE_PATH = r"c:\Users\megha\Desktop\nsclc_ga_svm\nsclc\media\models\nsclc_3d_cnn.h5"
    
    BATCH_SIZE = 4
    EPOCHS = 20
    TARGET_SHAPE = (64, 128, 128) # Depth, Height, Width
    
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    
    # Filter for valid labels (Adenocarcinoma & Squamous cell carcinoma)
    valid_classes = ["Adenocarcinoma", "Squamous cell carcinoma"]
    df = df[df['histology'].isin(valid_classes)].copy()
    
    # Construct full paths - fix path issues if any
    # The CSV has relative paths like /kaggle/input/..., we need to map them to local
    # We assume file names are mostly preserved.
    
    def get_local_path(row):
        filename = os.path.basename(row['image'])
        # Handle .nii.gz vs .nii
        if filename.endswith('.gz'):
            filename = filename[:-3]
        
        # Check local existence
        local_path = os.path.join(DATASET_DIR, filename)
        if os.path.exists(local_path):
            return local_path
        # Try replacing .nii.gz with .nii just in case logic above missed something (e.g. if file on disk is .nii but csv says .nii.gz)
        filename_nii = filename.replace(".nii.gz", ".nii")
        local_path_nii = os.path.join(DATASET_DIR, filename_nii)
        if os.path.exists(local_path_nii):
            return local_path_nii
            
        return None

    df['local_path'] = df.apply(get_local_path, axis=1)
    df = df.dropna(subset=['local_path'])
    
    if len(df) == 0:
        print("No valid image files found. Check dataset paths.")
        return

    print(f"Found {len(df)} valid samples.")
    
    # 2. Prepare Labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['histology'])
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    
    # 3. Train/Val Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)
    
    # 4. Create Generators
    train_gen = NiftiDataGenerator(
        train_df['local_path'].values,
        train_df['label_encoded'].values,
        batch_size=BATCH_SIZE,
        target_shape=TARGET_SHAPE,
        n_classes=num_classes,
        shuffle=True
    )
    
    val_gen = NiftiDataGenerator(
        val_df['local_path'].values,
        val_df['label_encoded'].values,
        batch_size=BATCH_SIZE,
        target_shape=TARGET_SHAPE,
        n_classes=num_classes,
        shuffle=False
    )
    
    # 5. Build Model
    model = build_3d_model(
        width=TARGET_SHAPE[2],
        height=TARGET_SHAPE[1],
        depth=TARGET_SHAPE[0],
        num_classes=num_classes
    )
    model.summary()
    
    # 6. Train
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    print("Training complete.")

if __name__ == "__main__":
    train_3d_model()
