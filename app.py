from flask import Flask, request, render_template, jsonify
import os
import torch
from generate_face import generate_face
from cyclegan import CycleGAN


app = Flask(__name__)


# Ensure upload directory exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Define device (use 'cuda' if GPU is available, otherwise 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load CycleGAN model
model = CycleGAN(device=device)
model.load_pretrained_weights('./pretrained_models/model_1.pth')  # Path to your pretrained weights


@app.route('/')
def index():
   """Render the main HTML page."""
   return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
   """Handle image upload and processing."""
   try:
       if 'file' not in request.files:
           return jsonify({"error": "No file part"})
      
       file = request.files['file']
       if file.filename == '':
           return jsonify({"error": "No selected file"})
      
       # Save the uploaded image
       image_path = os.path.join(UPLOAD_FOLDER, file.filename)
       file.save(image_path)


       # Get transformation parameters
       transform_type = request.form.get('type', '')  # 'age' or 'deage'
       age_level = request.form.get('age_level', '')


       # Validate inputs
       if not transform_type or not age_level.isdigit():
           return jsonify({"error": "Invalid input parameters"})
      
       age_level = int(age_level)


       # Process the image using CycleGAN
       processed_image_path = generate_face(image_path, transform_type, age_level, model)


       return jsonify({"output": processed_image_path})


   except Exception as e:
       return jsonify({"error": str(e)})


if __name__ == '__main__':
   app.run(debug=True
   )
