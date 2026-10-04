import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.ndimage import zoom

class NiftiDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Sequence for loading 3D NIfTI files on demand.
    """
    def __init__(self, image_paths, labels, batch_size=8, target_shape=(64, 128, 128), n_classes=2, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_shape = target_shape # (Depth, Height, Width)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_image_paths, batch_labels)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_labels):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.target_shape, 1))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(batch_image_paths):
            try:
                # Load and preprocess image
                volume = self.load_and_preprocess(path)
                X[i, ] = volume
                y[i] = batch_labels[i]
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Fallback: fill with zeros to avoid crashing batch
                X[i, ] = np.zeros((*self.target_shape, 1))
                y[i] = 0 

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def load_and_preprocess(self, path):
        # Load NIfTI
        img = nib.load(path)
        data = img.get_fdata()

        # Handle 4D data (e.g. time series or multiple channels) - take first channel
        if len(data.shape) == 4:
            data = data[..., 0]

        # Rotate to match standard orientation if needed (optional, depends on dataset)
        # data = np.rot90(data) 

        # Normalize intensity (Lung Window)
        # Common lung window: WL -600, WW 1500 -> [-1350, 150] roughly
        # For simplicity, we can clip to [-1000, 400] (HU units approx) then normalize
        min_hu = -1000
        max_hu = 400
        data = np.clip(data, min_hu, max_hu)
        data = (data - min_hu) / (max_hu - min_hu)
        
        # Resize volume
        # Current shape
        current_shape = data.shape
        
        # Calculate resize factors
        # Note: target_shape is (Depth, Height, Width), but data might be (H, W, D) or (D, H, W)
        # Usually NIfTI is (x, y, z). Let's assume (H, W, D) coming in mostly.
        # We need to map to our target (D, H, W) or (H, W, D).
        # Let's standardize on (Depth, Height, Width) for the model input.
        
        # If incoming is (X, Y, Z), we treat Z as depth.
        depth_factor = self.target_shape[0] / current_shape[2]
        height_factor = self.target_shape[1] / current_shape[0]
        width_factor = self.target_shape[2] / current_shape[1]
        
        factors = (height_factor, width_factor, depth_factor) 
        
        # Resize using spline interpolation (order=1 for speed)
        data = zoom(data, factors, order=1)
        
        # Transpose if necessary to match (Depth, Height, Width)
        # zoom returns (H_new, W_new, D_new). Model expects (D, H, W, C)
        data = np.transpose(data, (2, 0, 1)) 
        
        # Expand dims for channel
        data = np.expand_dims(data, axis=-1)
        
        return data
