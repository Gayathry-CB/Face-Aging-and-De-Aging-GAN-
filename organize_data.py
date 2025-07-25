import os
import shutil


# Define paths
dataset_dir = 'data/utkface/'  # Path to UTKFace dataset
young_faces_dir = 'data/young_faces/'  # Age < 30
middle_aged_faces_dir = 'data/middle_aged_faces/'  # 30 <= Age < 50 (Not used in CycleGAN)
old_faces_dir = 'data/old_faces/'  # Age >= 50


# Create directories if they don't exist
os.makedirs(young_faces_dir, exist_ok=True)
os.makedirs(middle_aged_faces_dir, exist_ok=True)
os.makedirs(old_faces_dir, exist_ok=True)


# Process images
for filename in os.listdir(dataset_dir):
   if filename.endswith('.jpg'):
       age = int(filename.split('_')[0])  # Extract age from filename


       # Move images based on age classification
       if age < 30:
           shutil.move(os.path.join(dataset_dir, filename), os.path.join(young_faces_dir, filename))
       elif 30 <= age < 50:
           shutil.move(os.path.join(dataset_dir, filename), os.path.join(middle_aged_faces_dir, filename))
       else:
           shutil.move(os.path.join(dataset_dir, filename), os.path.join(old_faces_dir, filename))


print("✅ Data successfully organized into young, middle-aged, and old categories!")