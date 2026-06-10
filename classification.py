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

from keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model
from keras.applications.efficientnet_v2 import preprocess_input

def load_img(image_path, label_path): 
    img = Image.open(image_path).convert("RGB")  # convert to grayscale
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

    canvas = Image.new("RGB", (size, size), (0, 0, 0)) # 3 channels black padding
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

def make_model(height, width, dim) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, dim)),
            data_augmentation,
            tf.keras.layers.Conv2D(64, 3, activation="relu"), 
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
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

def make_transfer_model(input_shape=(64, 64, 3)): 
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,  
        weights="imagenet",
        input_shape=input_shape,  
        pooling=None
    )

    base_model.trainable = False
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
    ])
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  
    print("EfficientNet layers:", len(base_model.layers))
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model, base_model

def train_model(model_path="classification_animals_model.keras", transfer=False): 
    X, y = load_dataset()
    print("images loaded")
    X = resize_img_padding(X, 164)
    print("images resized")
    
    if transfer:
        X = preprocess_input(X)
    else: 
        X /= 255.0

    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print(X_train.shape)
    print(X_test.shape)

    # imbalanced classes 
    class_weights = get_class_weights(y_train)
    print(class_weights)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    if transfer: 
        model, base_model = make_transfer_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            class_weight=class_weights,
            shuffle=True,
            verbose=2,
            callbacks=[early_stop]
        )

        base_model.trainable = True

        # Freeze most layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Recompile after changing trainable flags
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stop_stage2 = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            class_weight=class_weights,
            shuffle=True,
            verbose=2,
            callbacks=[
                early_stop_stage2,
            ]
        )

    else: 
        model = make_model(X_train.shape[1], X_train.shape[2], X_train.shape[3])   

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            class_weight=class_weights,
            verbose=2,
            shuffle=True,
            #callbacks=[early_stop]
        )

    print(model.summary())
    evaluate_model(model, X_test, y_test)
    model.save(model_path)

def load_model(model_path, train=False, transfer=True):
    X_train, y_train = load_dataset()
    X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")
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
    train_model("classification_transfer.keras", transfer=True)
    load_model("classification_transfer.keras")

# TODO's:
# try less training data samples and increase image size (to little GPU to do that now)
# try different model architectures (from scratch) like in the lecture
# try image preprocessing?? idk filtering and so on
# smaller batch size but increase image size 