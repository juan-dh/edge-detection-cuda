import os
import urllib.request
import tarfile
import numpy as np
import imageio
import shutil

# Paths
images_folder = "data/images"
extract_path = "data/stl10_binary"
download_path = "data/stl10_binary.tar.gz"
train_bin = os.path.join(extract_path, "train_X.bin")

# STL-10 dataset URL (binary)
url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# Download dataset if not already downloaded
if not os.path.exists(download_path):
    print("Downloading STL-10 dataset...")
    urllib.request.urlretrieve(url, download_path)
    print("Download completed.")
else:
    print("Download already exists, skipping.")

# Extract dataset if not already extracted
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path="data")
    print("Extraction completed.")
else:
    print("Dataset already extracted, skipping.")

# Create images folder if it doesn't exist
os.makedirs(images_folder, exist_ok=True)

# Function to save images from binary file
def save_images_from_bin(bin_path, folder):
    # Count already saved images
    existing_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    start_index = len(existing_files)
    with open(bin_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
        n_images = data.size // (3*96*96)
        data = data.reshape((n_images, 3, 96, 96))
        for i in range(start_index, n_images):
            img = np.transpose(data[i], (1,2,0))  # CxHxW -> HxWxC
            img = np.rot90(img, k=-1)             # rotate 90 degrees  clockwise
            filename = os.path.join(folder, f"img_{i:04d}.jpg")
            imageio.imwrite(filename, img)
    return n_images

# Save training images only if not already done

print("Saving training images...")
n_train = save_images_from_bin(train_bin, images_folder)
print(f"Saved {n_train} images in {images_folder}")


# Clean up temporary files (optional)

#if os.path.exists(download_path):
#    os.remove(download_path)
#if os.path.exists(extract_path):
#    shutil.rmtree(extract_path)
#print("Temporary files removed.")