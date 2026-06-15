"""
Classifies the animals into the 5 classes. 

Execute before (for preprocessing): 
1. build_labels.py
2. filter_label.py
3. filter_images.py
4. rebalance_splits.py
"""

from PIL import Image
import cv2
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import statistics

import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, concatenate
from keras.models import Model
from keras.applications.efficientnet_v2 import preprocess_input

def load_img(image_path, label_path, context_ratio=0.5):
    img = Image.open(image_path).convert("L")  # grayscale
    W, H = img.size
    
    cropped_images = []
    classes = []
    
    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split())
            
            if w * W < 5 or h * H < 5:
                continue
            
            # expand box by context_ratio
            w_exp = w * (1 + context_ratio)
            h_exp = h * (1 + context_ratio)
            
            x1 = max(0, int((xc - w_exp/2) * W))
            y1 = max(0, int((yc - h_exp/2) * H))
            x2 = min(W, int((xc + w_exp/2) * W))
            y2 = min(H, int((yc + h_exp/2) * H))
            
            crop = img.crop((x1, y1, x2, y2))
            cropped_images.append(crop)
            classes.append(int(cls))
    
    return cropped_images, classes

def load_img_rgb(image_path, label_path, rgb_path, context_ratio=0.5):
    img = Image.open(image_path).convert("L")  # grayscale
    rgb = Image.open(rgb_path).convert("RGB")
    W, H = img.size
    
    cropped_images = []
    cropped_rgbs = []
    classes = []
    
    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split())
            
            if w * W < 5 or h * H < 5:
                continue
            
            # expand box by context_ratio
            w_exp = w * (1 + context_ratio)
            h_exp = h * (1 + context_ratio)
            
            x1 = max(0, int((xc - w_exp/2) * W))
            y1 = max(0, int((yc - h_exp/2) * H))
            x2 = min(W, int((xc + w_exp/2) * W))
            y2 = min(H, int((yc + h_exp/2) * H))
            
            crop = img.crop((x1, y1, x2, y2))
            crop_rgb = rgb.crop((x1, y1, x2, y2))
            cropped_images.append(crop)
            cropped_rgbs.append(crop_rgb)
            classes.append(int(cls))
    
    return cropped_images, cropped_rgbs, classes

def load_dataset_rgb(image_path="images_filtered/train", label_path="labels_filtered/train", rgb_path="/home/azureuser/cloudfiles/code/Users/s2410929006/CVI/rgb_filtered/rgb_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)
    rgb_dir = Path(rgb_path)

    X = []
    X_rgb = []
    Y = []
    for img_path in images_dir.glob("*"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        rgb_path = rgb_dir / f"{img_path.stem}.jpg"

        if lbl_path.exists() and rgb_path.exists():
            x, x_rgb, y = load_img_rgb(img_path, lbl_path, rgb_path)
            X.extend(x)
            X_rgb.extend(x_rgb)
            Y.extend(y)
    
    return X, X_rgb, Y

def load_dataset(image_path="images_filtered/train", label_path="labels_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)

    X = []
    Y = []
    flight_ids = []
    for img_path in images_dir.glob("*"):
        lbl_path = labels_dir / f"{img_path.stem}.txt" 

        if lbl_path.exists():
            x, y = load_img(img_path, lbl_path)
            X.extend(x)
            Y.extend(y)
            for _ in range(len(y)): 
                flight_ids.append(img_path.stem.split('_')[0])
    
    return X, Y, flight_ids

# dont change format of image, add padding if bounding boxes not a square
def resize_img_padding(X, size=128): 
    x = np.array([np.array(resize_with_padding(img, size)) for img in X], dtype=np.float32)

    return x

def resize_with_padding(img, size=128):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))

    if img.mode == "RGB":
        fill = (128, 128, 128)
    else:
        fill = 128

    canvas = Image.new(img.mode, (size, size), fill)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas

def mask_rgb(X):
    masked_rgbs = []

    for img in X:
        thermal = img[...,0]
        thr = np.percentile(thermal, 95)
        mask = thermal > thr # only keep the brightest places

        kernel = np.ones((7,7), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (11,11), 0)

        # apply to rgb
        rgb = img[..., 1:]
        rgb_masked = rgb * mask[...,None]
        masked_rgbs.append(rgb_masked)
    
    return np.array(masked_rgbs, dtype=np.float32)


def get_class_weights(y): 
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))

def make_robust_generalizer(height, width, dim) -> tf.keras.Model:
    image_input = Input(shape=(height, width, dim))
    
    # 1. Moderate augmentation to diversify without erasing species signatures
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.20),
        tf.keras.layers.RandomZoom(0.10),
    ])
    
    x = data_augmentation(image_input)
    
    # Block 1 - Expanding feature map count to capture thermal gradients
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Global Pool to preserve spatial invariance across different flights
    x = GlobalAveragePooling2D()(x) 
    
    # Expanded dense capacity so it has room to think
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  
    
    output = Dense(5, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=image_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), # Slight bump to kick-start weights
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def make_model(width, height, dim) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), # Animals can face any direction relative to drones horizontal_and_vertical
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        #tf.keras.layers.RandomTranslation(0.1,0.1)
    ])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, dim)),
            data_augmentation,
            tf.keras.layers.Conv2D(64, 3, activation="relu"), 
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation="relu"),  #256
            BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def make_small_cnn(input_shape=(128, 128, 1), num_classes=5):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1,0.1),
        
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def transfer_model(input_shape=(128, 128, 3)):
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,  
        weights="imagenet",
        input_shape=input_shape,  
        pooling=None
    )

    base_model.trainable=False
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
    ])
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

def create_3class_dataset(y):
    y_new = []

    for cls in y:
        if cls in [0, 1, 2]:
            y_new.append(0)  # all deer merged
        elif cls == 3:
            y_new.append(1)  # wild boar
        elif cls == 4:
            y_new.append(2)  # hybrid pig

    return np.array(y_new, dtype=np.int32)

def calc_flight_stats(flight_ids, y, X, split="train"):
    print(f"{split} flight ids + statistic")
    flight_ids = np.array(flight_ids, dtype=np.int32)

    for flight in np.unique(flight_ids):
        for species in np.unique(y):
            images = [
                img
                for img, f, y in zip(X, flight_ids, y)
                if f == flight and y == species
            ]

            if len(images) > 20:
                print(f"for flight {flight} and species {species}: ")
                brightnesses = [np.array(img, dtype=np.int32).mean() for img in images]
                print(f"min: {min(brightnesses)}, max: {max(brightnesses)}, mean: {statistics.mean(brightnesses)}, median: {statistics.median(brightnesses)}")
                contrasts = [np.array(img, dtype=np.int32).std() for img in images]
                print(f"min: {min(contrasts)}, max: {max(contrasts)}, mean: {statistics.mean(contrasts)}, median: {statistics.median(contrasts)}")
                ranges = [np.array(img, dtype=np.int32).max() - np.array(img, dtype=np.int32).min() for img in images]
                print(f"min: {min(ranges)}, max: {max(ranges)}, mean: {statistics.mean(ranges)}, median: {statistics.median(ranges)}")

def oversampling(X, y):
    # Find the size of your largest class to balance up to it
    max_class_size = max(np.bincount(y))
    X_resampled = []
    y_resampled = []

    for class_idx in np.unique(y):
        X_class = X[y == class_idx]
        y_class = y[y == class_idx]
        
        # Oversample minority classes with replacement
        X_upsampled, y_upsampled = resample(
            X_class, y_class,
            replace=True,
            n_samples=max_class_size,
            random_state=42
        )
        X_resampled.extend(X_upsampled)
        y_resampled.extend(y_upsampled)

    return X_resampled, y_resampled

def combine_rgb_grayscale(gray, rgb):
    if(len(gray.shape) < 4):
        gray = np.expand_dims(gray, axis=-1)
    X = np.concatenate([gray, rgb], axis=-1)
    print(X.shape)
    return X

def model_combined():
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    rgb_input = keras.Input((128,128,3))

    rgb_branch = base_model(rgb_input, training=False)
    rgb_branch = keras.layers.GlobalAveragePooling2D()(rgb_branch)

    thermal_input = Input((128,128,1))

    thermal_branch = Conv2D(32,3,padding="same",activation="relu")(thermal_input)
    thermal_branch = BatchNormalization()(thermal_branch)
    thermal_branch = MaxPooling2D()(thermal_branch)

    thermal_branch = Conv2D(64,3,padding="same",activation="relu")(thermal_branch)
    thermal_branch = BatchNormalization()(thermal_branch)
    thermal_branch = MaxPooling2D()(thermal_branch)

    thermal_branch = Conv2D(128,3,padding="same",activation="relu")(thermal_branch)
    thermal_branch = BatchNormalization()(thermal_branch)

    thermal_branch = GlobalAveragePooling2D()(thermal_branch)

    thermal_branch = Dense(64, activation="relu")(thermal_branch)

    x = keras.layers.concatenate([
        rgb_branch,
        thermal_branch
    ])

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(5, activation="softmax")(x)

    model = keras.Model(
        [rgb_input, thermal_input],
        out
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, concatenate
from keras.models import Model

def make_mid_fusion_drone_model(height=128, width=128) -> tf.keras.Model:
    """
    Separates Thermal shape extraction from RGB texture extraction, 
    fusing representations late in the network topology.
    """
    # -----------------------------------------------------------------
    # BRANCH 1: RGB Texture Pathway (Leveraging Transfer Learning)
    # -----------------------------------------------------------------
    rgb_input = Input(shape=(height, width, 3), name="rgb_input")
    
    # Moderate augmentation tailored for RGB environments
    rgb_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.10),
    ])(rgb_input)
    
    # Load EfficientNet pre-trained on ImageNet for rich texture maps
    rgb_base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        pooling=None
    )
    rgb_base.trainable = False  # Keep features frozen to protect minority gradients
    
    rgb_features = rgb_base(rgb_aug, training=False)
    rgb_vector = GlobalAveragePooling2D()(rgb_features)
    rgb_vector = Dense(128, activation="relu")(rgb_vector)
    rgb_vector = BatchNormalization()(rgb_vector)

    # -----------------------------------------------------------------
    # BRANCH 2: Thermal Hot-Spot Structural Pathway (Custom CNN)
    # -----------------------------------------------------------------
    thermal_input = Input(shape=(height, width, 1), name="thermal_input")
    
    # Geometric transformations for flight adjustments
    thermal_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.05),
    ])(thermal_input)
    
    t_layer = Conv2D(32, 3, padding='same', activation="relu")(thermal_aug)
    t_layer = BatchNormalization()(t_layer)
    t_layer = MaxPooling2D()(t_layer)
    
    t_layer = Conv2D(64, 3, padding='same', activation="relu")(t_layer)
    t_layer = BatchNormalization()(t_layer)
    t_layer = MaxPooling2D()(t_layer)
    
    t_layer = Conv2D(128, 3, padding='same', activation="relu")(t_layer)
    t_layer = BatchNormalization()(t_layer)
    
    thermal_vector = GlobalAveragePooling2D()(t_layer)
    thermal_vector = Dense(64, activation="relu")(thermal_vector)
    thermal_vector = BatchNormalization()(thermal_vector)

    # -----------------------------------------------------------------
    # CONCATENATION AND CROSS-MODAL FUSION HEAD
    # -----------------------------------------------------------------
    # Merge the 128-dim RGB vector with the 64-dim Thermal vector
    fused_embedding = concatenate([rgb_vector, thermal_vector])
    
    x = Dense(128, activation="relu")(fused_embedding)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # High dropout prevents dominant class memorization
    
    outputs = Dense(5, activation="softmax", name="species_output")(x)

    # Compile model expecting a dual input stream list
    model = Model(inputs=[rgb_input, thermal_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# execute once - takes around 15 min.
def preprocess_and_save_data():
    X_train, X_train_rgb, y_train = load_dataset_rgb()
    X_val, X_val_rgb, y_val = load_dataset_rgb("images_filtered/val", "labels_filtered/val", "/home/azureuser/cloudfiles/code/Users/s2410929006/CVI/rgb_filtered/rgb_filtered/val")
    X_test, X_test_rgb, y_test = load_dataset_rgb("images_filtered/test", "labels_filtered/test", "/home/azureuser/cloudfiles/code/Users/s2410929006/CVI/rgb_filtered/rgb_filtered/test")
    print("images loaded")
    print(len(X_train), len(X_train_rgb), len(y_train))
    print(len(X_val), len(X_val_rgb), len(y_val))
    print(len(X_test), len(X_test_rgb), len(y_test))

    X_train = resize_img_padding(X_train, 128)
    X_train_rgb = resize_img_padding(X_train_rgb, 128)
    X_val = resize_img_padding(X_val, 128)
    X_val_rgb = resize_img_padding(X_val_rgb, 128)
    X_test = resize_img_padding(X_test, 128)
    X_test_rgb = resize_img_padding(X_test_rgb, 128)

    print(X_train.shape, X_train_rgb.shape)
    print(X_val.shape, X_val_rgb.shape)
    print(X_test.shape, X_test_rgb.shape)

    X_train = combine_rgb_grayscale(X_train, X_train_rgb)
    X_val = combine_rgb_grayscale(X_val, X_val_rgb)
    X_test = combine_rgb_grayscale(X_test, X_test_rgb)

    print("images resized")

    # RGB channels
    X_train[...,1:] /= 255.0
    X_val[...,1:] /= 255.0
    X_test[...,1:] /= 255.0

    # Thermal channel
    thermal_mean = X_train[...,0].mean()
    thermal_std = X_train[...,0].std()

    X_train[...,0] = (X_train[...,0] - thermal_mean) / thermal_std
    X_val[...,0] = (X_val[...,0] - thermal_mean) / thermal_std
    X_test[...,0] = (X_test[...,0] - thermal_mean) / thermal_std

    # save the preprocessed data
    np.savez_compressed(
        "train.npz",
        X=X_train.astype(np.float32),
        y=np.array(y_train, dtype=np.int32),
        thermal_mean=thermal_mean,
        thermal_std=thermal_std
    )

    np.savez_compressed(
        "val.npz",
        X=X_val.astype(np.float32),
        y=np.array(y_val, dtype=np.int32)
    )

    np.savez_compressed(
        "test.npz",
        X=X_test.astype(np.float32),
        y=np.array(y_test, dtype=np.int32)
    )

def make_optimized_drone_cnn(width, height, dim) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"), 
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.10),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(height, width, dim)),
        data_augmentation,
        
        # Block 1: Capture fine-grained boundary edges
        tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"), 
        BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        # Block 2: Expand features intermediate scale
        tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
        BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        
        # Block 3: Deep structural features (Narrower filters prevent background smearing)
        tf.keras.layers.Conv2D(128, 3, padding='same', activation="relu"),
        BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation="relu"),
        BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense Generalization Layers
        tf.keras.layers.Dense(64, activation="relu"), 
        BatchNormalization(),
        tf.keras.layers.Dropout(0.4), # Elevated slightly to curb the 90% overfit
        tf.keras.layers.Dense(5, activation="softmax"),
    ])

    # Lower learning rate + explicit label smoothing to handle oversampled profiles safely
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

def make_mid_fusion_drone_model(height=128, width=128) -> tf.keras.Model:
    """
    Separates Thermal shape extraction from RGB texture extraction, 
    fusing representations late in the network topology.
    """
    # -----------------------------------------------------------------
    # BRANCH 1: RGB Texture Pathway (Leveraging Transfer Learning)
    # -----------------------------------------------------------------
    rgb_input = Input(shape=(height, width, 3), name="rgb_input")
    
    # Moderate augmentation tailored for RGB environments
    rgb_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.10),
    ])(rgb_input)
    
    # Load EfficientNet pre-trained on ImageNet for rich texture maps
    rgb_base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        pooling=None
    )
    rgb_base.trainable = False  # Keep features frozen to protect minority gradients
    
    rgb_features = rgb_base(rgb_aug, training=False)
    rgb_vector = GlobalAveragePooling2D()(rgb_features)
    rgb_vector = Dense(128, activation="relu")(rgb_vector)
    rgb_vector = BatchNormalization()(rgb_vector)

    # -----------------------------------------------------------------
    # BRANCH 2: Thermal Hot-Spot Structural Pathway (Custom CNN)
    # -----------------------------------------------------------------
    thermal_input = Input(shape=(height, width, 1), name="thermal_input")
    
    # Geometric transformations for flight adjustments
    thermal_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.05),
    ])(thermal_input)
    
    t_layer = Conv2D(32, 3, padding='same', activation="relu")(thermal_aug)
    t_layer = BatchNormalization()(t_layer)
    t_layer = MaxPooling2D()(t_layer)
    
    t_layer = Conv2D(64, 3, padding='same', activation="relu")(t_layer)
    t_layer = BatchNormalization()(t_layer)
    t_layer = MaxPooling2D()(t_layer)
    
    t_layer = Conv2D(128, 3, padding='same', activation="relu")(t_layer)
    t_layer = BatchNormalization()(t_layer)
    
    thermal_vector = GlobalAveragePooling2D()(t_layer)
    thermal_vector = Dense(64, activation="relu")(thermal_vector)
    thermal_vector = BatchNormalization()(thermal_vector)

    # -----------------------------------------------------------------
    # CONCATENATION AND CROSS-MODAL FUSION HEAD
    # -----------------------------------------------------------------
    # Merge the 128-dim RGB vector with the 64-dim Thermal vector
    fused_embedding = concatenate([rgb_vector, thermal_vector])
    
    x = Dense(128, activation="relu")(fused_embedding)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # High dropout prevents dominant class memorization
    
    outputs = Dense(5, activation="softmax", name="species_output")(x)

    # Compile model expecting a dual input stream list
    model = Model(inputs=[rgb_input, thermal_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def mask_rgb_2(X_combined):
    masked_combined = np.copy(X_combined)
    for i in range(len(X_combined)):
        thermal = X_combined[i, ..., 0]
        # Secure top 5% brightest pixels reliably before any normalization 
        thr = np.percentile(thermal, 95)
        mask = (thermal > thr).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)

        # Apply mask directly to the RGB layers safely
        masked_combined[i, ..., 1:] = X_combined[i, ..., 1:] * mask[..., None]
    return masked_combined

def train_model(model_path="classification_animals_model.keras"): 
    # Oversamling
    #X_train, y_train = oversampling(X_train, y_train)

    train = np.load("train.npz")
    X_train = train["X"]
    y_train = train["y"]

    val = np.load("val.npz")
    X_val = val["X"]
    y_val = val["y"]

    test = np.load("test.npz")
    X_test = test["X"]
    y_test = test["y"]

    X_train = mask_rgb_2(X_train)
    X_val = mask_rgb_2(X_val)
    X_test = mask_rgb_2(X_test)

    X_train_thermal = X_train[:, :, :, 0:1]
    X_val_thermal = X_val[:, :, :, 0:1]
    X_test_thermal = X_test[:, :, :, 0:1]

    X_train_rgb = X_train[:, :, :, 1:]
    X_val_rgb = X_val[:, :, :, 1:]
    X_test_rgb = X_test[:, :, :, 1:]

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print(X_train_thermal.shape)
    print(X_test_thermal.shape)
    print(X_train_rgb.shape)
    print(X_test_rgb.shape)


    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    model = make_mid_fusion_drone_model()

    model.fit(
        [X_train_rgb, X_train_thermal],
        y_train,
        validation_data=([X_val_rgb, X_val_thermal], y_val),
        epochs=20,
        batch_size=32,
        verbose=2,
        shuffle=True,
        callbacks=[early_stop]
    )

    print(model.summary())
    evaluate_model(model, [X_val_rgb, X_val_thermal], y_val, "Val")
    evaluate_model(model, [X_test_rgb, X_test_thermal], y_test)
    model.save(model_path)

    #class_weights = get_class_weights(y_train)
    #print("Applying Class Weights:", class_weights)

def load_model(model_path, train=False):
    X_train, y_train,_ = load_dataset()
    X_val, y_val,_ = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test,_ = load_dataset("images_filtered/test", "labels_filtered/test")

    widths = []
    w_train = []
    w_val = []
    w_test = []
    heights = []
    h_train = []
    h_val = []
    h_test = []
    X = X_train + X_val + X_test
    for img in X:
        widths.append(img.size[0])
        heights.append(img.size[1])

    for img in X_train:
        w_train.append(img.size[0])
        h_train.append(img.size[1])
    for img in X_val:
        w_val.append(img.size[0])
        h_val.append(img.size[1])
    for img in X_test:
        w_test.append(img.size[0])
        h_test.append(img.size[1])
    print(min(widths), max(widths), statistics.mean(widths), statistics.median(widths))
    print(min(heights), max(heights), statistics.mean(heights), statistics.median(heights))
    print("\n\nX_train:")
    print(min(w_train), max(w_train), statistics.mean(w_train), statistics.median(w_train))
    print(min(h_train), max(h_train), statistics.mean(h_train), statistics.median(h_train))
    print("\n\nX_val:")
    print(min(w_val), max(w_val), statistics.mean(w_val), statistics.median(w_val))
    print(min(h_val), max(h_val), statistics.mean(h_val), statistics.median(h_val))
    print("\n\nX_test:")
    print(min(w_test), max(w_test), statistics.mean(w_test), statistics.median(w_test))
    print(min(h_test), max(h_test), statistics.mean(h_test), statistics.median(h_test))

    print("images loaded")
    X_train = resize_img_padding(X_train, 128)
    X_val = resize_img_padding(X_val, 128)
    X_test = resize_img_padding(X_test, 128)
    print("images resized")
    # need additional dim for keras input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    loaded_model = keras.saving.load_model(model_path)

    if train: 
        # imbalanced classes 
        class_weights = get_class_weights(y_train)
        print(class_weights)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )

        loaded_model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            class_weight=class_weights,
            verbose=2,
            shuffle=True,
            callbacks=[early_stop]
        )

    evaluate_model(loaded_model, X_train, y_train, "Train")
    evaluate_model(loaded_model, X_val, y_val, "Val")
    evaluate_model(loaded_model, X_test, y_test, "Test")

def evaluate_model(model, X, y, split="Test"):
    loss, acc = model.evaluate(X, y)
    print(f"{split} loss:", loss)
    print(f"{split} accuracy:", acc)

    # print a confusion matrix
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y, y_pred_classes)
    print(cm)

    balanced_acc = balanced_accuracy_score(y, y_pred_classes)
    print("Balanced accuracy:", balanced_acc)
    print(classification_report(y, y_pred_classes, digits=3))

if __name__ == "__main__":
    #preprocess_and_save_data()
    train_model("species_class_models/combined_branched_mid_fusion.keras")

# TODO's:
# different model architectures (from scratch) like in the lecture
# image preprocessing?? idk filtering and so on
# higher context ratio (at loading the images)