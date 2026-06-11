"""
Classifies the animals into the 5 classes. 

Execute before (for preprocessing): 
1. build_labels.py
2. filter_label.py
3. filter_images.py
4. rebalance_splits.py
"""

from PIL import Image
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import statistics

from keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, concatenate
from keras.models import Model
from keras.applications.efficientnet_v2 import preprocess_input

def load_img(image_path, label_path): 
    img = Image.open(image_path).convert("L")  # convert to grayscale
    W, H = img.size

    cropped_images = []
    classes = []

    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split()) # convert them to float, no need to import motion

            # exclude samples that are less than 5 pixels wide or high -> too small
            if w * W < 5 or h * H < 5:
                continue

            x1 = int((xc - w/2) * W)
            y1 = int((yc - h/2) * H)
            x2 = int((xc + w/2) * W)
            y2 = int((yc + h/2) * H)

            crop = img.crop((x1, y1, x2, y2))
            cropped_images.append(crop)
            classes.append(int(cls))
    
    return cropped_images, classes

def load_dataset(image_path="images_filtered/train", label_path="labels_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)

    X = []
    Y = []
    for img_path in images_dir.glob("*"):
        lbl_path = labels_dir / f"{img_path.stem}.txt" 
        
        if lbl_path.exists():
            x, y = load_img(img_path, lbl_path)
            X.extend(x)
            Y.extend(y)
    
    return X, Y

def load_img_multimodal(image_path, label_path): 
    img = Image.open(image_path).convert("L")
    W, H = img.size

    cropped_images = []
    classes = []
    meta_features = [] # To capture scale data [w, h]

    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split())

            if w * W < 5 or h * H < 5:
                continue

            x1 = int((xc - w/2) * W)
            y1 = int((yc - h/2) * H)
            x2 = int((xc + w/2) * W)
            y2 = int((yc + h/2) * H)

            crop = img.crop((x1, y1, x2, y2))
            cropped_images.append(crop)
            classes.append(int(cls))
            meta_features.append([w, h]) # Keep the metadata frame values
    
    return cropped_images, classes, meta_features

def load_dataset_multimodal(image_path="images_filtered/train", label_path="labels_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)

    X, Y, M = [], [], []
    for img_path in images_dir.glob("*"):
        lbl_path = labels_dir / f"{img_path.stem}.txt" 
        
        if lbl_path.exists():
            x, y, m = load_img_multimodal(img_path, lbl_path)
            X.extend(x)
            Y.extend(y)
            M.extend(m)
    
    return X, Y, M

# resize bc for training they all need to have the same size
def resize_img(X_train, X_val, X_test, size=(128, 128)):
    # convert to correct input format for tensorflow
    x_train = np.array([np.array(img.resize(size)) for img in X_train], dtype=np.float32)
    x_val = np.array([np.array(img.resize(size)) for img in X_val], dtype=np.float32)
    x_test = np.array([np.array(img.resize(size)) for img in X_test], dtype=np.float32)

    return x_train, x_val, x_test

# dont change format of image, add padding if bounding boxes not a square
def resize_img_padding(X, size=128): 
    x = np.array([np.array(resize_with_padding(img, size)) for img in X], dtype=np.float32)

    return x

def resize_with_padding(img, size=128):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))

    canvas = Image.new("L", (size, size), (128))
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas

def get_class_weights(y): 
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input

def make_model(height, width, dim) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"), # Animals can face any direction relative to drones
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        Input(shape=(height, width, dim)),
        data_augmentation,
        
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        # Reduced dense layer capacity to force general feature representation
        Dense(128, activation="relu"), 
        BatchNormalization(),
        Dropout(0.5), # Increased dropout to heavily penalize memorization
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def make_multimodal_model(height, width, dim) -> tf.keras.Model:
    # --- Branch A: Pixel Image Processor ---
    image_input = Input(shape=(height, width, dim), name="image_input")
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.20), # Increased to completely scramble flight patterns
    ])
    
    x = data_augmentation(image_input)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    # --- Branch B: Scale Bounding Box Processor ---
    meta_input = Input(shape=(2,), name="meta_input")
    y = Dense(32, activation="relu")(meta_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation="relu")(y)

    # --- Fusion Core ---
    combined = concatenate([x, y])
    z = Dense(64, activation="relu")(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z) # Aggressive dropout to penalize background memorization
    
    output = Dense(5, activation="softmax", name="output")(z)
    
    model = Model(inputs=[image_input, meta_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), # Low learning rate for validation stability
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(model_path="classification_animals_model.keras"): 
    X_train, y_train = load_dataset()
    X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")
    print("images loaded")

    X_train = resize_img_padding(X_train, 128)
    X_val = resize_img_padding(X_val, 128)
    X_test = resize_img_padding(X_test, 128)

    # need additional dim for keras input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    print("images resized")

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Oversamling
    # Find the size of your largest class to balance up to it
    max_class_size = max(np.bincount(y_train))
    X_train_resampled = []
    y_train_resampled = []

    for class_idx in np.unique(y_train):
        X_class = X_train[y_train == class_idx]
        y_class = y_train[y_train == class_idx]
        
        # Oversample minority classes with replacement
        X_upsampled, y_upsampled = resample(
            X_class, y_class,
            replace=True,
            n_samples=max_class_size,
            random_state=42
        )
        X_train_resampled.extend(X_upsampled)
        y_train_resampled.extend(y_upsampled)

    X_train = np.array(X_train_resampled, dtype=np.float32)
    y_train = np.array(y_train_resampled, dtype=np.int32)

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print(X_train.shape)
    print(X_test.shape)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    model = make_model(X_train.shape[1], X_train.shape[2], X_train.shape[3])   

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=2,
        shuffle=True,
        #callbacks=[early_stop]
    )

    print(model.summary())
    evaluate_model(model, X_val, y_val)
    evaluate_model(model, X_test, y_test)
    model.save(model_path)

def train_model_multimodal(model_path="classification_multimodal.keras"): 
    X_train, y_train, M_train = load_dataset_multimodal()
    X_val, y_val, M_val = load_dataset_multimodal("images_filtered/val", "labels_filtered/val")
    X_test, y_test, M_test = load_dataset_multimodal("images_filtered/test", "labels_filtered/test")
    
    # Run processing and channel expanding
    X_train = resize_img_padding(X_train, 128)[..., np.newaxis] / 255.0
    X_val = resize_img_padding(X_val, 128)[..., np.newaxis] / 255.0
    X_test = resize_img_padding(X_test, 128)[..., np.newaxis] / 255.0
    
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    M_train = np.array(M_train, dtype=np.float32)
    M_val = np.array(M_val, dtype=np.float32)
    M_test = np.array(M_test, dtype=np.float32)

    # --- Synchronized Multimodal Oversampling ---
    max_class_size = max(np.bincount(y_train))
    X_resampled, y_resampled, M_resampled = [], [], []

    for class_idx in np.unique(y_train):
        indices = np.where(y_train == class_idx)[0]
        # Resample matching indices across images and shape metadata
        sampled_indices = np.random.choice(indices, size=max_class_size, replace=True)
        
        X_resampled.extend(X_train[sampled_indices])
        y_resampled.extend(y_train[sampled_indices])
        M_resampled.extend(M_train[sampled_indices])

    X_train = np.array(X_resampled, dtype=np.float32)
    y_train = np.array(y_resampled, dtype=np.int32)
    M_train = np.array(M_resampled, dtype=np.float32)

    model = make_multimodal_model(128, 128, 1)
    
    # Train passing inputs as a multi-key dictionary
    model.fit(
        x={"image_input": X_train, "meta_input": M_train},
        y=y_train,
        validation_data=({"image_input": X_val, "meta_input": M_val}, y_val),
        epochs=25,
        batch_size=32, # 32 provides stabler batches than 16 for gradient steps
        shuffle=True,
        verbose=2
    )
    
    # Evaluation
    loss, acc = model.evaluate({"image_input": X_test, "meta_input": M_test}, y_test)
    print(f"Test Accuracy: {acc:.3f}")

def load_model(model_path, train=False, transfer=True):
    X_train, y_train = load_dataset()
    X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")

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

    if transfer:           
        # convert to rgb format for transfer learning
        X_train = np.repeat(X_train, 3, axis=-1)
        X_val   = np.repeat(X_val, 3, axis=-1)
        X_test  = np.repeat(X_test, 3, axis=-1)
    
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)
        X_test = preprocess_input(X_test)
    else: 
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
    #load_model("classification_base.keras")
    train_model_multimodal("classification_multi.keras")
    #load_model("classification_base16.keras")

# TODO's:
# try less training data samples and increase image size (to little GPU to do that now)
# try different model architectures (from scratch) like in the lecture
# try image preprocessing?? idk filtering and so on
# smaller batch size but increase image size 