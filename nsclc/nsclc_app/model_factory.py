import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

def build_model(num_classes=2, input_shape=(224, 224, 3)):
    """
    Builds a MobileNetV2 based model for Transfer Learning.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path, class_names):
    """
    Predicts the class of a single image.
    """
    try:
        img = Image.open(image_path).convert('L').convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        preds = model.predict(img_array)
        score = tf.nn.softmax(preds[0]) # Depending on activation, this might be redundant if final is softmax
        
        # If model output is softmax, preds[0] sums to 1 directly
        # If we use binary crossentropy with sigmoid, it's different.
        # Here we used categorical_crossentropy with softmax.
        
        class_idx = np.argmax(preds[0])
        confidence = np.max(preds[0])
        
        return class_names[class_idx], float(confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

def build_3d_model(width=128, height=128, depth=64, num_classes=2):
    """
    Builds a 3D Convolutional Neural Network.
    """
    inputs = tf.keras.Input((depth, height, width, 1))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="3dcnn")
    
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )
    
    return model

def build_ann_model(input_dim, num_classes=2):
    """
    Builds a simple Artificial Neural Network (ANN) for tabular data.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
        
    outputs = Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model
