from pathlib import Path
import shutil

def filter_images(label_path="labels_filtered", image_path="images_filtered", output_path="images_filtered2"):
    dirs = ["/train/", "/test/", "/val/"]

    for dir in dirs: 
        labels_dir = Path(label_path + dir)
        images_dir = Path(image_path + dir)
        output_dir = Path(output_path + dir)
        output_dir.mkdir(exist_ok=True)

        valid_stems = {f.stem for f in labels_dir.glob("*.txt")} # without file endings

        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.stem in valid_stems:
                shutil.copy2(img_file, output_dir / img_file.name)

if __name__ == "__main__":
    filter_images()