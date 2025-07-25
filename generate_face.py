
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os


# ----------------------------------------
# Generator Network
# ----------------------------------------
class Generator(nn.Module):
   def __init__(self):
       super(Generator, self).__init__()


       self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
       self.relu = nn.ReLU(inplace=True)


       # Downsampling
       self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
       self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)


    # Upsampling
       self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
       self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)


       self.conv_out = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
       self.tanh = nn.Tanh()


   def forward(self, x):
       x = self.relu(self.conv1(x))
       x = self.relu(self.conv2(x))
       x = self.relu(self.conv3(x))
       x = self.relu(self.deconv1(x))
       x = self.relu(self.deconv2(x))
       return self.tanh(self.conv_out(x))


# ----------------------------------------
# CycleGAN Model
# ----------------------------------------
class CycleGAN(nn.Module):
   def __init__(self, device=None):
       super(CycleGAN, self).__init__()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device


       self.generator_A_to_B = Generator().to(self.device)  # Young → Old
       self.generator_B_to_A = Generator().to(self.device)  # Old → Young


   def forward(self, x):
       return self.generator_A_to_B(x), self.generator_B_to_A(x)


   def load_pretrained_weights(self, path):
       try:
           checkpoint = torch.load(path, map_location=self.device)
           print(f"\n🧠 Checkpoint loaded: {path}")


           # Loading only the model weights
           model_weights = checkpoint['model_state_dict']
           self.generator_A_to_B.load_state_dict(model_weights, strict=False)
           self.generator_B_to_A.load_state_dict(model_weights, strict=False)


           print("✅ Pretrained weights loaded successfully.\n")
       except Exception as e:
           print(f"❌ Error loading weights from {path}: {e}")




# ----------------------------------------
# Preprocessing and Generation
# ----------------------------------------


# Transformation pipeline
transform = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
def denormalize(tensor):
   """Denormalize the tensor to [0, 1] for float image output."""
   tensor = tensor.clone()
   tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
   return tensor.clamp(0, 1)








def visualize_channels(tensor):
   """Visualize each RGB channel separately."""
   tensor = tensor.squeeze(0).cpu().detach()  # Remove batch dimension
   red_channel = tensor[0, :, :].cpu().numpy()
   green_channel = tensor[1, :, :].cpu().numpy()
   blue_channel = tensor[2, :, :].cpu().numpy()


   # Plot each channel
   fig, axes = plt.subplots(1, 3, figsize=(12, 4))
   axes[0].imshow(red_channel, cmap='Reds')
   axes[0].set_title("Red Channel")
   axes[1].imshow(green_channel, cmap='Greens')
   axes[1].set_title("Green Channel")
   axes[2].imshow(blue_channel, cmap='Blues')
   axes[2].set_title("Blue Channel")


   plt.show()


def visualize_tensor(tensor):
   """Visualize tensor as a heatmap to inspect values."""
   tensor = tensor.squeeze(0).cpu().detach().numpy()
   plt.imshow(tensor.transpose(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
   plt.colorbar()
   plt.show()






def generate_face(image_path, transform_type, age_level, model, output_folder='static/output'):
   try:
       # Load image and preprocess
       image = Image.open(image_path).convert("RGB")
       input_tensor = transform(image).unsqueeze(0).to(model.device)


       model.eval()
       with torch.no_grad():
           if transform_type == 'age':
               output_tensor = model.generator_A_to_B(input_tensor)
           elif transform_type == 'deage':
               output_tensor = model.generator_B_to_A(input_tensor)
           else:
               raise ValueError("Invalid transform_type. Use 'age' or 'deage'.")


       # Debugging: print shape and min/max of output tensor
       print("🔍 Output tensor shape:", output_tensor.shape)
       print("🔍 Output tensor min:", output_tensor.min().item(), "max:", output_tensor.max().item())


# Visualize tensor values before denormalization
       print(f"Before denormalization - min: {output_tensor.min().item()} max: {output_tensor.max().item()}")
       visualize_tensor(output_tensor)  # Visualize tensor as heatmap


       # Denormalize
       output_tensor = denormalize(output_tensor.squeeze(0).detach().cpu())


       # Debugging: After denormalization
       print(f"After denormalization - min: {output_tensor.min().item()} max: {output_tensor.max().item()}")
       visualize_tensor(output_tensor)  # Visualize tensor after denormalization


       # Convert tensor to PIL image
       output_image = to_pil_image(output_tensor)


       # Check if the image has visible content
       output_image.show()  # Open image to visually check


       # ✅ Save image
       os.makedirs(output_folder, exist_ok=True)
       output_image_path = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
       output_image.save(output_image_path)


       print(f"✅ Image saved at: {output_image_path}")
       return output_image_path


   except Exception as e:
       return f"Error during face generation: {e}"




# ----------------------------------------
# Expose model for import
# ----------------------------------------
model = CycleGAN()
model.load_pretrained_weights('checkpoints/cyclegan_epoch_190.pth')


# Only for internal test run
if __name__ == "__main__":
   # Run on an actual image for testing
   test_image_path = 'static/uploads/test_1.jpg'  # Put a test image here
   transform_type = 'age'
   age_level = 0  # Not used


   output_path = generate_face(test_image_path, transform_type, age_level, model, output_folder='static/output')
   print(f"🖼️ Processed image saved at: {output_path}")





