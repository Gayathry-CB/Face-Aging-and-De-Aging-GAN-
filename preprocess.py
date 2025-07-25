import os
import cv2
import numpy as np
from tqdm import tqdm


# Directories for input and output
input_dirs = {
   'young': 'data/young_faces/',
   'middle_aged': 'data/middle_aged_faces/',
   'old': 'data/old_faces/'
}


output_dirs = {
   'young': 'data/processed_young/',
   'middle_aged': 'data/processed_middle_aged/',
   'old': 'data/processed_old/'
}


# Create output directories if they don't exist
for dir_path in output_dirs.values():
   os.makedirs(dir_path, exist_ok=True)


# Preprocessing function
def preprocess_images(input_dir, output_dir, img_size=(256,256)):
   for filename in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
       if filename.endswith('.jpg'):
           img_path = os.path.join(input_dir, filename)
           img = cv2.imread(img_path)  # Load image
          
           # Check if image is loaded properly
           if img is None:
               print(f"Error loading image: {filename}")
               continue
          
           # Resize image to the specified size (256x256)
           img = cv2.resize(img, img_size)


           # Check the image dimensions after resizing
           print(f"Resized image {filename} shape: {img.shape}")
          
           # Normalize the image to [-1, 1] range
           img = (img / 127.5) - 1


           # Save the processed image as a .npy file
           np.save(os.path.join(output_dir, filename.replace('.jpg', '.npy')), img)


# Preprocess all domains (Young, Middle-aged, Old)
for key in input_dirs:
   print(f"Starting preprocessing for {key} faces...")
   preprocess_images(input_dirs[key], output_dirs[key])


print("✅ Image preprocessing complete!")
