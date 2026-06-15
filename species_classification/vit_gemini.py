import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from PIL import Image
import cv2
import statistics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D,
                          Dense, Dropout, BatchNormalization, Input, LayerNormalization, 
                          MultiHeadAttention, Reshape)
from keras.models import Model

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================

def load_img_rgb(image_path, label_path, rgb_path, context_ratio=0.5):
    img = Image.open(image_path).convert("L")  # Thermal / Grayscale
    rgb = Image.open(rgb_path).convert("RGB")   # Color background
    W, H = img.size
    
    cropped_images = []
    cropped_rgbs = []
    classes = []
    
    with open(label_path) as f:
        for annotation in f:
            cls, xc, yc, w, h, _ = map(float, annotation.split())
            
            if w * W < 5 or h * H < 5:
                continue
            
            # Expand bounding box slightly for environmental context
            w_exp = w * (1 + context_ratio)
            h_exp = h * (1 + context_ratio)
            
            x1 = max(0, int((xc - w_exp/2) * W))
            y1 = max(0, int((yc - h_exp/2) * H))
            x2 = min(W, int((xc + w_exp/2) * W))
            y2 = min(H, int((yc + h_exp/2) * H))
            
            cropped_images.append(img.crop((x1, y1, x2, y2)))
            cropped_rgbs.append(rgb.crop((x1, y1, x2, y2)))
            classes.append(int(cls))
            
    return cropped_images, cropped_rgbs, classes

def load_dataset_rgb(image_path="images_filtered/train", label_path="labels_filtered/train", rgb_path="rgb_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)
    rgb_dir = Path(rgb_path)

    X, X_rgb, Y = [], [], []
    for img_path in images_dir.glob("*"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        r_path = rgb_dir / f"{img_path.stem}.jpg"

        if lbl_path.exists() and r_path.exists():
            x, x_rgb, y = load_img_rgb(img_path, lbl_path, r_path)
            X.extend(x)
            X_rgb.extend(x_rgb)
            Y.extend(y)
    
    return X, X_rgb, Y

def resize_img_padding(X, size=128): 
    return np.array([np.array(resize_with_padding(img, size)) for img in X], dtype=np.float32)

def resize_with_padding(img, size=128):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))

    fill = (128, 128, 128) if img.mode == "RGB" else 128
    canvas = Image.new(img.mode, (size, size), fill)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas

def combine_rgb_grayscale(gray, rgb):
    if len(gray.shape) < 4:
        gray = np.expand_dims(gray, axis=-1)
    return np.concatenate([gray, rgb], axis=-1)

# ==========================================
# 2. THERMAL GUIDED IMAGE MASKING
# ==========================================

def mask_rgb(X_combined):
    """
    Applies a dilation mask generated via raw pixel thresholding on the 
    thermal channel directly to the matching RGB frames before data scaling.
    """
    masked_combined = np.copy(X_combined)
    for i in range(len(X_combined)):
        thermal = X_combined[i, ..., 0]
        thr = np.percentile(thermal, 95)
        mask = (thermal > thr).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)

        # Map attention filters onto structural background
        masked_combined[i, ..., 1:] = X_combined[i, ..., 1:] * mask[..., None]
    return masked_combined

def get_class_weights(y): 
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

# ==========================================
# 3. HYBRID VISION TRANSFORMER MODEL Build
# ==========================================

def vit_transformer_block(inputs, num_heads=4, key_dim=64, ff_dim=256):
    """Standard Transformer Encoder block with LayerNorm and Attention"""
    # Self Attention Path
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = attention_output + inputs
    
    # MLP Feed Forward Path
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation="gelu")(y)
    y = Dropout(0.3)(y)
    y = Dense(inputs.shape[-1])(y)
    return x + y

def make_hybrid_vit_drone_model(height=128, width=128, dim=4) -> tf.keras.Model:
    inputs = Input(shape=(height, width, dim))
    
    # 1. Augmentation Strategy 
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.15)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    # 2. CNN Backbone (Extracts Inductive Local Features / Signatures)
    x = Conv2D(32, 3, padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x) # 64x64x32
    
    x = Conv2D(64, 3, padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x) # 32x32x64
    
    x = Conv2D(128, 3, padding='same', activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x) # 16x16x128

    # 3. Spatial Tokenization Bridge
    # Flattens 16x16 feature maps into sequences of 256 structural patches
    num_tokens = (height // 8) * (width // 8)
    projection_dim = 128
    tokens = Reshape((num_tokens, projection_dim))(x)
    
    # 4. Attention Pipeline 
    tokens = vit_transformer_block(tokens, num_heads=4, key_dim=64, ff_dim=256)
    tokens = vit_transformer_block(tokens, num_heads=4, key_dim=64, ff_dim=256)
    
    # 5. Global Token Pooling & Classification Output Head
    global_representation = GlobalAveragePooling1D()(tokens)
    
    x = Dense(64, activation="relu")(global_representation)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(5, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for stable attention training
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ==========================================
# 4. EXECUTION PIPELINE CONTROL
# ==========================================

def preprocess_and_save_data():
    print("⏳ Starting Multi-Modal pipeline load...")
    X_train, X_train_rgb, y_train = load_dataset_rgb()
    X_val, X_val_rgb, y_val = load_dataset_rgb("images_filtered/val", "labels_filtered/val", "rgb_filtered/val")
    X_test, X_test_rgb, y_test = load_dataset_rgb("images_filtered/test", "labels_filtered/test", "rgb_filtered/test")
    
    X_train, X_train_rgb = resize_img_padding(X_train, 128), resize_img_padding(X_train_rgb, 128)
    X_val, X_val_rgb = resize_img_padding(X_val, 128), resize_img_padding(X_val_rgb, 128)
    X_test, X_test_rgb = resize_img_padding(X_test, 128), resize_img_padding(X_test_rgb, 128)

    X_train = combine_rgb_grayscale(X_train, X_train_rgb)
    X_val = combine_rgb_grayscale(X_val, X_val_rgb)
    X_test = combine_rgb_grayscale(X_test, X_test_rgb)

    # Apply masking sequence directly on RAW pixel spaces
    X_train = mask_rgb(X_train)
    X_val = mask_rgb(X_val)
    X_test = mask_rgb(X_test)

    # Scale RGB layers safely (Channels 1, 2, 3)
    X_train[..., 1:] /= 255.0
    X_val[..., 1:] /= 255.0
    X_test[..., 1:] /= 255.0

    # Z-Score Standardize Thermal Layer independently (Channel 0)
    thermal_mean = X_train[..., 0].mean()
    thermal_std = X_train[..., 0].std()
    X_train[..., 0] = (X_train[..., 0] - thermal_mean) / thermal_std
    X_val[..., 0] = (X_val[..., 0] - thermal_mean) / thermal_std
    X_test[..., 0] = (X_test[..., 0] - thermal_mean) / thermal_std

    np.savez_compressed("train.npz", X=X_train.astype(np.float32), y=np.array(y_train, dtype=np.int32))
    np.savez_compressed("val.npz", X=X_val.astype(np.float32), y=np.array(y_val, dtype=np.int32))
    np.savez_compressed("test.npz", X=X_test.astype(np.float32), y=np.array(y_test, dtype=np.int32))
    print("✅ Preprocessing completed. Datasets saved safely.")

def train_model(): 
    # Run absolute preprocessing fresh 
    #preprocess_and_save_data()

    train = np.load("train.npz")
    X_train, y_train = train["X"], train["y"]

    val = np.load("val.npz")
    X_val, y_val = val["X"], val["y"]

    test = np.load("test.npz")
    X_test, y_test = test["X"], test["y"]

    class_weights = get_class_weights(y_train)
    print("\n⚖️ Applying Loss-Weights balancing for natural data distributions:\n", class_weights)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    )

    # Instantiate the Hybrid ViT network
    model = make_hybrid_vit_drone_model(X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(model.summary())

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        verbose=2,
        shuffle=True,
        class_weight=class_weights,  
        callbacks=[early_stop]
    )

    # Evaluate against un-contamination protocols
    evaluate_model(model, X_val, y_val, "Val")
    evaluate_model(model, X_test, y_test, "Test")
    model.save("hybrid_transformer_drone_model.keras")

def evaluate_model(model, X, y, split="Test"):
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\n📊 {split} Performance Summary Metrics:")
    print(f"{split} Overall Loss: {loss:.4f} | {split} Accuracy: {acc:.4f}")

    y_pred = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred_classes))
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y, y_pred_classes):.4f}")
    print("\nDetailed Classification Breakdown:")
    print(classification_report(y, y_pred_classes, digits=3))

if __name__ == "__main__":
    train_model()