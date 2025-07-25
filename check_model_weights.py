import torch

# Path to your model checkpoint
checkpoint_path = "checkpoints/latest_model.pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

print("Checkpoint keys:", checkpoint.keys())  # See available keys

