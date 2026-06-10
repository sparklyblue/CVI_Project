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
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h))

    canvas = Image.new("L", (size, size), 0)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(img, (x, y))

    return canvas

def get_class_weights(y): 
    classes = np.unique(y.numpy())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.numpy()
    )

    return dict(zip(classes, weights))

def make_model(width, height) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, 1)),
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

if __name__ == "__main__":
    X_train, y_train = load_dataset()
    #X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    #X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")
    print("images loaded")
    X_train = resize_img_padding(X_train, 128)
    #X_val = resize_img_padding(X_val, 128)
    #X_test = resize_img_padding(X_test, 128)
    print("images resized")
    # need additional dim for keras input
    X_train = X_train[..., np.newaxis]
    #X_val = X_val[..., np.newaxis]
    #X_test = X_test[..., np.newaxis]

    # convert to rgb format for transfer learning
    X_train = np.repeat(X_train, 3, axis=-1)
    #X_val   = np.repeat(X_val, 3, axis=-1)
    #X_test  = np.repeat(X_test, 3, axis=-1)
    
    X_train = preprocess_input(X_train)
    #X_val = preprocess_input(X_val)
    #X_test = preprocess_input(X_test)

    #X_train /= 255.0
    #X_val /= 255.0
    #X_test /= 255.0

    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    #y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    #y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    print(np.unique(y_train.numpy(), return_counts=True))
    print(np.unique(y_test.numpy(), return_counts=True))

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

    #model = make_model(X_train.shape[1], X_train.shape[2])
    model = make_transfer_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        #validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        verbose=2,
        shuffle=True,
        callbacks=[early_stop]
    )

    print(model.summary())
    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    # print a confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    balanced_acc = balanced_accuracy_score(y_test, y_pred_classes)
    print("Balanced accuracy:", balanced_acc)
    print(classification_report(y_test, y_pred_classes, digits=3))
    model.save("classification_animals_model.keras")
