import torch
from cyclegan import CycleGAN


# Function to load the CycleGAN model
def load_cycle_gan_model(model_path):
   """
   Load the CycleGAN model from the provided path.


   Args:
   - model_path (str): The path to the pre-trained model weights.


   Returns:
   - model (CycleGAN): The loaded CycleGAN model.
   """
   # Initialize the CycleGAN model
   model = CycleGAN()
  
   # Load the pretrained model weights
   model.load_pretrained_weights(model_path)
  
   # Set the model to evaluation mode
   model.eval()
  
   return model


# Example usage (this is just for reference, it's not necessary for the main app to have this part)
if __name__ == "__main__":
   model_path = './pretrained_models/model_1.pth'  # Path to the pretrained model
   model = load_cycle_gan_model(model_path)
   print("Model loaded successfully!")


