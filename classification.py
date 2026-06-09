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
from sklearn.metrics import confusion_matrix

from keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

def load_img(image_path, label_path): 
    img = Image.open(image_path).convert("L")  # convert to grayscale
    W, H = img.size

    cropped_images = []
    classes = []

    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split()) # convert them to float, no need to import motion
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

    for img, label in zip(sorted(images_dir.iterdir()), sorted(labels_dir.iterdir())):
        x, y = load_img(img, label)
        X.extend(x)
        Y.extend(y)
    
    return X, Y

# resize bc for training they all need to have the same size
def resize_img(X_train, X_val, X_test, size=(128, 128)):
    X = X_train + X_val + X_test

    # convert to correct input format for tensorflow
    x_train = np.array([np.array(img.resize(size)) for img in X_train], dtype=np.float32)
    x_val = np.array([np.array(img.resize(size)) for img in X_val], dtype=np.float32)
    x_test = np.array([np.array(img.resize(size)) for img in X_test], dtype=np.float32)

    return x_train, x_val, x_test

# do i want to do that?!
def augment_data():
    pass

def make_model(width, height) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, 1)),
            tf.keras.layers.Conv2D(64, 3, activation="relu"), 
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32),
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

    base_model.trainable = True
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)  
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

def evaluate(X_test, y_test):
    pass

if __name__ == "__main__":
    X_train, y_train = load_dataset()
    X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")
    print("images loaded")
    X_train, X_val, X_test = resize_img(X_train, X_val, X_test)
    print("images resized")
    # need additional dim for keras input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    # convert to rgb format for transfer learning
    X_train = np.repeat(X_train, 3, axis=-1)
    X_val   = np.repeat(X_val, 3, axis=-1)
    X_test  = np.repeat(X_test, 3, axis=-1)

    print(X_train.shape)
    print(np.unique_counts(y_train))

    #model = make_model(X_train.shape[1], X_train.shape[2])
    model = make_transfer_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=2,
        shuffle=True,
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
