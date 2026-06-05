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
            classes.append(cls)
    
    return cropped_images, classes

def load_dataset(image_path="images_filtered/train", label_path="labels_filtered/train"): 
    labels_dir = Path(label_path)
    images_dir = Path(image_path)

    X = []
    Y = []

    for img, label in zip(images_dir.iterdir(), labels_dir.iterdir()):
        x, y = load_img(img, label)
        X.extend(x)
        Y.extend(y)
    
    return X, Y

# resize bc for training they all need to have the same size
def resize_img(X_train, X_val, X_test):
    X = X_train + X_val + X_test
    max_width = max(img.width for img in X)
    max_height = max(img.height for img in X)

    X_train = [img.resize((max_width, max_height)) for img in X_train]
    X_val = [img.resize((max_width, max_height)) for img in X_val]
    X_test = [img.resize((max_width, max_height)) for img in X_test]

    return X_train, X_val, X_test

# do i want to do that?!
def augment_data():
    pass

def train_model(X_train, y_train): 
    pass

def evaluate(X_test, y_test):
    pass

if __name__ == "__main__":
    X_train, y_train = load_dataset()
    X_val, y_val = load_dataset("images_filtered/val", "labels_filtered/val")
    X_test, y_test = load_dataset("images_filtered/test", "labels_filtered/test")

    X_train, X_val, X_test = resize_img(X_train, X_val, X_test)
    print(len(X_train), len(y_train))
    print(len(X_val), len(y_val))    
    print(len(X_test), len(y_test))   