import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_factory import build_model
# import matplotlib.pyplot as plt

def train_model(epochs=10, batch_size=32, img_size=(224, 224)):
    # Paths
    base_dir = r"c:\Users\megha\Desktop\nsclc_ga_svm\dataset"
    model_save_path = r"c:\Users\megha\Desktop\nsclc_ga_svm\nsclc\media\models\nsclc_cnn.h5"
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(base_dir):
        print(f"Dataset directory not found: {base_dir}")
        return

    # Hyperparameters
    BATCH_SIZE = batch_size
    IMG_SIZE = img_size
    EPOCHS = epochs
    
    # Data Augmentation and Loading
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Build Model
    model = build_model(num_classes=train_generator.num_classes)
    
    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Save Model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Optional: Save training plot
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig('training_plot.png')

if __name__ == "__main__":
    train_model()
