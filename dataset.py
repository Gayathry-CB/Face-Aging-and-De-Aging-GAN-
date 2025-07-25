import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FaceAgingDataset(Dataset):
   def __init__(self, data_dir_A, data_dir_B, transform=None):
       self.data_dir_A = data_dir_A
       self.data_dir_B = data_dir_B
       self.transform = transform
       self.image_files_A = [f for f in os.listdir(data_dir_A) if f.endswith('.npy')]
       self.image_files_B = [f for f in os.listdir(data_dir_B) if f.endswith('.npy')]


   def __len__(self):
       return len(self.image_files_A)


   def __getitem__(self, idx):
       img_A_path = os.path.join(self.data_dir_A, self.image_files_A[idx])
       img_B_path = os.path.join(self.data_dir_B, self.image_files_B[idx])
       img_A = np.load(img_A_path)  # Load .npy file for image A
       img_B = np.load(img_B_path)  # Load .npy file for image B


       # Convert numpy array (H, W, C) to tensor (C, H, W)
       img_A = torch.tensor(img_A, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
       img_B = torch.tensor(img_B, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)


       if self.transform:
           img_A = self.transform(img_A)
           img_B = self.transform(img_B)


       return img_A, img_B


# Define transformations (resize, normalization)
transform = transforms.Compose([
   transforms.Resize((64, 64)),
   transforms.Normalize(mean=[-1, -1, -1], std=[1, 1, 1])  # Already normalized in preprocess step
])


# Load Young, Middle-Aged, and Old datasets
young_dataset = FaceAgingDataset("data/processed_young", "data/processed_old", transform=transform)
middle_aged_dataset = FaceAgingDataset("data/processed_middle_aged", "data/processed_old", transform=transform)
old_dataset = FaceAgingDataset("data/processed_old", "data/processed_young", transform=transform)


# Create DataLoaders
batch_size = 16
young_loader = DataLoader(young_dataset, batch_size=batch_size, shuffle=True)
middle_aged_loader = DataLoader(middle_aged_dataset, batch_size=batch_size, shuffle=True)
old_loader = DataLoader(old_dataset, batch_size=batch_size, shuffle=True)


print(f"✅ DataLoader is ready! Young: {len(young_dataset)} images, Middle-Aged: {len(middle_aged_dataset)} images, Old: {len(old_dataset)} images")
