from pathlib import Path
import shutil

def filter_images(label_path="labels_filtered", image_path="images_thermal/images", output_path="images_filtered"):
    dirs = ["/train/", "/test/", "/val/"]
    
    valid_stems = set()
    for dir in dirs: 
        labels_dir = Path(label_path + dir)
        for f in labels_dir.glob("*.txt"):
            valid_stems.add(f.stem)

    for dir in dirs: 
        images_dir = Path(image_path + dir)
        output_dir = Path(output_path + dir)
        output_dir.mkdir(exist_ok=True)

        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.stem in valid_stems:
                shutil.copy2(img_file, output_dir / img_file.name)

def cmp_img():
    test_imgs = Path("rgb_filtered/test/")
    stems = {f.stem for f in test_imgs.glob("*.jpg")}

    dir = Path("images_filtered/test/")
    for f in dir.glob("*.jpg"):
        if f.stem not in stems:
            print(f.stem)

if __name__ == "__main__":
    #+filter_images(image_path="images_rgb/images", output_path="rgb_filtered")
    cmp_img()