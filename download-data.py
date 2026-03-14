import os
import urllib.request
import tarfile
import numpy as np
import imageio
import shutil

# Paths
stl10_images_folder = "data/stl10_images"
uscsipi_images_folder = "data/uscsipi_images"
stl10_extract_path = "data/stl10_binary"
uscsipi_extract_path = "data/uscsipi_files"
stl10_download_path = "data/stl10_binary.tar.gz"
aerials_download_path = "data/aerials.tar.gz"
misc_download_path = "data/misc.tar.gz"
train_bin = os.path.join(stl10_extract_path, "train_X.bin")

# STL-10 dataset URL (binary)
stl10_url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# USCSIPI dataset URLs
aerials_url = "https://sipi.usc.edu/database/aerials.tar.gz"
misc_url = "https://sipi.usc.edu/database/misc.tar.gz"

# Download dataset if not already downloaded
if not os.path.exists(stl10_download_path):
    print("Downloading STL-10 dataset...")
    urllib.request.urlretrieve(stl10_url, stl10_download_path)
    print("Download completed.")
else:
    print("STL-10 download already exists, skipping.")

# Extract dataset if not already extracted
if not os.path.exists(stl10_extract_path):
    print("Extracting STL-10 dataset...")
    with tarfile.open(stl10_download_path, "r:gz") as tar:
        tar.extractall(path="data")
    print("Extraction completed.")
else:
    print("STL-10 dataset already extracted, skipping.")

# Download USCSIPI datasets if not already downloaded
if not os.path.exists(aerials_download_path):
    print("Downloading USCSIPI aerials dataset...")
    urllib.request.urlretrieve(aerials_url, aerials_download_path)
    print("Aerials download completed.")
else:
    print("Aerials download already exists, skipping.")

if not os.path.exists(misc_download_path):
    print("Downloading USCSIPI misc dataset...")
    urllib.request.urlretrieve(misc_url, misc_download_path)
    print("Misc download completed.")
else:
    print("Misc download already exists, skipping.")

# Extract USCSIPI datasets if not already extracted
uscsipi_extract_base = "data/uscsipi_files"
os.makedirs(uscsipi_extract_base, exist_ok=True)

if not os.path.exists(os.path.join(uscsipi_extract_base, "aerials")):
    print("Extracting USCSIPI aerials dataset...")
    with tarfile.open(aerials_download_path, "r:gz") as tar:
        tar.extractall(path=uscsipi_extract_base)
    print("Aerials extraction completed.")
else:
    print("Aerials dataset already extracted, skipping.")

if not os.path.exists(os.path.join(uscsipi_extract_base, "misc")):
    print("Extracting USCSIPI misc dataset...")
    with tarfile.open(misc_download_path, "r:gz") as tar:
        tar.extractall(path=uscsipi_extract_base)
    print("Misc extraction completed.")
else:
    print("Misc dataset already extracted, skipping.")

# Create images folders if they don't exist
os.makedirs(stl10_images_folder, exist_ok=True)
os.makedirs(uscsipi_images_folder, exist_ok=True)

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

# Function to copy images from USCSIPI folders
def copy_images_from_folder(source_folder, dest_folder):
    # Count already saved images
    existing_files = [f for f in os.listdir(dest_folder) if f.endswith(".jpg") or f.endswith(".png")]
    start_index = len(existing_files)
    
    count = 0
    for root, dirs, files in os.walk(source_folder):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                src_path = os.path.join(root, file)
                dest_filename = os.path.join(dest_folder, f"img_{start_index + count:04d}.jpg")
                
                # Read and convert to JPG
                img = imageio.imread(src_path)
                # Convert grayscale to RGB if needed
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 4:  # RGBA to RGB
                    img = img[:, :, :3]
                
                imageio.imwrite(dest_filename, img)
                count += 1
    
    return count

# Save STL-10 training images
print("\nProcessing STL-10 dataset...")
print("Saving training images...")
n_train = save_images_from_bin(train_bin, stl10_images_folder)
print(f"Saved {n_train} images in {stl10_images_folder}")

# Save USCSIPI images
print("\nProcessing USCSIPI dataset...")
aerials_folder = os.path.join(uscsipi_extract_base, "aerials")
misc_folder = os.path.join(uscsipi_extract_base, "misc")

if os.path.exists(aerials_folder):
    print("Saving USCSIPI aerials images...")
    n_aerials = copy_images_from_folder(aerials_folder, uscsipi_images_folder)
    print(f"Saved {n_aerials} aerials images")

if os.path.exists(misc_folder):
    print("Saving USCSIPI misc images...")
    n_misc = copy_images_from_folder(misc_folder, uscsipi_images_folder)
    print(f"Saved {n_misc} misc images")

print(f"\nTotal images in {uscsipi_images_folder}: {len([f for f in os.listdir(uscsipi_images_folder) if f.endswith(('.jpg', '.png'))])}")


# Clean up temporary files (optional)

#if os.path.exists(stl10_download_path):
#    os.remove(stl10_download_path)
#if os.path.exists(extract_path):
#    shutil.rmtree(extract_path)
#print("Temporary files removed.")