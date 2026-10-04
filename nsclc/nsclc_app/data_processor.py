import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from django.conf import settings

def process_nifti_to_2d(nifti_path, target_size=(224, 224)):
    """
    Loads a NIfTI file, extracts the most informative slice (max variance), and resizes it.
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Handle 4D data
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
            
        # Find slice with max standard deviation (most information/contrast)
        best_slice = None
        max_std = -1
        
        # Iterate over depth
        for i in range(data.shape[2]):
            s = data[:, :, i]
            std = s.std()
            if std > max_std:
                max_std = std
                best_slice = s
                
        if best_slice is None or max_std < 5: # Skip if image is mostly empty
            return None
            
        # Normalize to 0-255
        slice_2d = ((best_slice - best_slice.min()) / (best_slice.max() - best_slice.min() + 1e-8)) * 255
        slice_2d = slice_2d.astype(np.uint8)
        
        # Convert to Grayscale then RGB to match standard 3-channel input
        pil_img = Image.fromarray(slice_2d).convert('L').convert('RGB')
        pil_img = pil_img.resize(target_size)
        
        return pil_img
    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")
        return None

def prepare_dataset(base_dir, output_dir, metadata_path):
    """
    Reads metadata, processes files, and organizes them into class folders.
    Now handles mixed NIfTI (cancerous) and JPG (normal) data.
    """
    cancerous_dir = os.path.join(base_dir, 'cancerous')
    normal_dir = os.path.join(base_dir, 'normal')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Process Cancerous (NIfTI) ---
    if os.path.exists(cancerous_dir) and os.path.exists(metadata_path):
        print(f"Processing cancerous data from {cancerous_dir}...")
        df = pd.read_csv(metadata_path)
        
        for _, row in df.iterrows():
            label = row['histology']
            if pd.isna(label):
                continue
            clean_label = label.lower().replace(" ", "_")
            
            filename = os.path.basename(row['image'])
            if filename.endswith('.gz'):
                filename = filename[:-3]
            filename_nii = filename.replace('.nii.gz', '.nii')
            
            local_path = os.path.join(cancerous_dir, filename_nii)
            
            if not os.path.exists(local_path):
                continue
                
            class_dir = os.path.join(output_dir, clean_label)
            os.makedirs(class_dir, exist_ok=True)
            
            output_filename = f"{row['patient']}.png"
            output_path = os.path.join(class_dir, output_filename)
            
            # Reprocess even if exists to apply new slicing logic
            # if os.path.exists(output_path):
            #    continue
            
            img = process_nifti_to_2d(local_path)
            if img:
                img.save(output_path)
    else:
        print("Cancerous data directory or metadata not found.")

    # --- Process Normal (JPG) ---
    if os.path.exists(normal_dir):
        print(f"Processing normal data from {normal_dir}...")
        normal_output_dir = os.path.join(output_dir, 'normal')
        os.makedirs(normal_output_dir, exist_ok=True)
        
        for fname in os.listdir(normal_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(normal_dir, fname)
                dst_path = os.path.join(normal_output_dir, fname)
                
                # Check if already processed (or just copy/resize)
                # Reprocess to ensure grayscale
                try:
                    with Image.open(src_path) as img:
                        # Force Grayscale then RGB
                        img = img.convert('L').convert('RGB')
                        img = img.resize((224, 224))
                        img.save(dst_path)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

    print("Data processing complete.")

if __name__ == "__main__":
    BASE_DIR = r"c:\Users\megha\Desktop\nsclc_ga_svm\dataset"
    OUTPUT_DIR = r"c:\Users\megha\Desktop\nsclc_ga_svm\nsclc\media\dataset_2d"
    METADATA_PATH = r"c:\Users\megha\Desktop\nsclc_ga_svm\dataset\cancerous\metadata.csv"
    
    prepare_dataset(BASE_DIR, OUTPUT_DIR, METADATA_PATH)
