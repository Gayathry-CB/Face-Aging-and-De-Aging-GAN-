import torch
import torch.nn as nn


# ----------------------------------------
# Generator Network
# ----------------------------------------


class Generator(nn.Module):
   def __init__(self):
       super(Generator, self).__init__()


       # Initial convolution layer
       self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
       self.relu = nn.ReLU(inplace=True)


       # Downsampling layers
       self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
       self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)


       # Upsampling layers to get to 64x64
       self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
       self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)


       # Output layer to get 64x64 image
       self.conv_out = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
       self.tanh = nn.Tanh()


   def forward(self, x):
       x = self.relu(self.conv1(x))
       x = self.relu(self.conv2(x))
       x = self.relu(self.conv3(x))


       # Upsample to 64x64
       x = self.relu(self.deconv1(x))
       x = self.relu(self.deconv2(x))


       return self.tanh(self.conv_out(x))  # Output image with size (64, 64)






# ----------------------------------------
# Discriminator Network
# ----------------------------------------
# In cyclegan.py, inside the Discriminator class


class Discriminator(nn.Module):
   def __init__(self):
       super(Discriminator, self).__init__()


       self.relu = nn.ReLU(inplace=True)
       self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
       self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
       self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
       self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)


       # Final fully connected layer for scalar output
       self.fc = nn.Linear(512 * 4 * 4, 1)  # Output size 1
       self.sigmoid = nn.Sigmoid()


   def forward(self, x):
       x = self.relu(self.conv1(x))
       x = self.relu(self.conv2(x))
       x = self.relu(self.conv3(x))
       x = self.relu(self.conv4(x))


       # Flatten for the fully connected layer
       x = x.view(x.size(0), -1)  # Flatten the tensor
       x = self.fc(x)  # Fully connected layer
       return self.sigmoid(x)  # Scalar output indicating real/fake








# ----------------------------------------
# CycleGAN Model
# ----------------------------------------
class CycleGAN(nn.Module):
   def __init__(self, device=None):
       super(CycleGAN, self).__init__()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device


       # Use updated generator and discriminator
       self.generator_A_to_B = Generator().to(self.device)  # Young → Old
       self.generator_B_to_A = Generator().to(self.device)  # Old → Young


       self.discriminator_A = Discriminator().to(self.device)  # Discriminator for real/ fake Young images
       self.discriminator_B = Discriminator().to(self.device)  # Discriminator for real/ fake Old images


   def forward(self, x):
       return self.generator_A_to_B(x), self.generator_B_to_A(x)


   def load_pretrained_weights(self, path):
       try:
           checkpoint = torch.load(path, map_location=self.device)
           print(f"\n🧠 Checkpoint loaded: {path}")
           print("🔍 Keys preview (first 5):", list(checkpoint.keys())[:5])


           # Load both generators and discriminators
           self.generator_A_to_B.load_state_dict(checkpoint['generator_A_to_B'], strict=False)
           self.generator_B_to_A.load_state_dict(checkpoint['generator_B_to_A'], strict=False)
           self.discriminator_A.load_state_dict(checkpoint['discriminator_A'], strict=False)
           self.discriminator_B.load_state_dict(checkpoint['discriminator_B'], strict=False)


           print("✅ Pretrained weights loaded successfully.\n")
       except Exception as e:
           print(f"❌ Error loading weights from {path}: {e}")







