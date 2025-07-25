import torch
from PIL import Image
import numpy as np
from cyclegan import CycleGAN


# Function to preprocess the image
def preprocess_image(image):
   image = image.convert("RGB")  # Ensure the image is in RGB format
   image = image.resize((256, 256))  # Resize image to the model's expected size
   image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
   image_tensor = torch.tensor(image_array).float().unsqueeze(0).permute(0, 3, 1, 2)  # Convert to tensor
   return image_tensor


# Function to process the face (either aging or de-aging)
def process_face(image, model, age_input=True, age_level=50):
   # Preprocess the image for model input
   processed_image = preprocess_image(image)
  
   # Run the image through the model
   with torch.no_grad():
       output_image = model(processed_image, age_input=age_input, age_level=age_level)
  
   # Convert the model's output tensor to a PIL Image
   output_image = output_image.squeeze().permute(1, 2, 0).numpy() * 255
   output_image = Image.fromarray(output_image.astype('uint8'))
  
   return output_image


# Inference function that handles the model and image processing
def run_inference(input_image_path, output_image_path, model, age_input=True, age_level=50):
   # Load the input image
   image = Image.open(input_image_path)
  
   # Process the image (age or de-age based on the input)
   processed_image = process_face(image, model, age_input, age_level)
  
   # Save the processed image
   processed_image.save(output_image_path)
  
   return output_image_path


# Example usage
if __name__ == "__main__":
   # Initialize the CycleGAN model and load pretrained weights
   model = CycleGAN()
   model.load_pretrained_weights('./pretrained_models/model_1.pth')  # Path to the pretrained model
  
   # Example image input and output paths
   input_image_path = './input_images/sample.jpg'  # Path to the image to process
   output_image_path = './output_images/processed_sample.jpg'  # Path where the output will be saved
  
   # Run the inference to age the face (age_input=True), or de-age it (age_input=False)
   age_level = 75  # Adjust this as needed
   result_image_path = run_inference(input_image_path, output_image_path, model, age_input=True, age_level=age_level)
  
   print(f"Processed image saved at: {result_image_path}")
