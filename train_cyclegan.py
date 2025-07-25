import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from cyclegan import CycleGAN
from models.dataset import young_loader, middle_aged_loader, old_loader


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CycleGAN(device=device)


# Define loss functions
criterion_GAN = nn.MSELoss()  # Adversarial loss
criterion_cycle = nn.L1Loss()  # Cycle-consistency loss


# Optimizers for both generators and discriminators
lr = 0.0002
optimizer_G = optim.Adam(list(model.generator_A_to_B.parameters()) + list(model.generator_B_to_A.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(model.discriminator_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(model.discriminator_B.parameters(), lr=lr, betas=(0.5, 0.999))


# Training loop
epochs = 200  # Set number of epochs
for epoch in range(epochs):
   for i, (real_A, real_B) in enumerate(young_loader):  # Change to appropriate dataloader
       # Move data to device
       real_A = real_A.to(device)
       real_B = real_B.to(device)


       batch_size = real_A.size(0)  # Get the batch size from the input data


       # Adversarial ground truths
       valid = torch.ones(batch_size, 1).to(device)  # Shape: [batch_size, 1]
       fake_valid = torch.zeros(batch_size, 1).to(device)  # For fake images


       # Train Generators (minimize adversarial loss, cycle consistency loss)
       optimizer_G.zero_grad()


       # Generator A→B and B→A
       fake_B = model.generator_A_to_B(real_A)
       fake_A = model.generator_B_to_A(real_B)


       # Adversarial loss for generators
       loss_GAN_A_to_B = criterion_GAN(model.discriminator_B(fake_B), valid)
       loss_GAN_B_to_A = criterion_GAN(model.discriminator_A(fake_A), valid)


       # Cycle consistency loss
       recovered_A = model.generator_B_to_A(fake_B)
       recovered_B = model.generator_A_to_B(fake_A)
       loss_cycle_A = criterion_cycle(recovered_A, real_A)
       loss_cycle_B = criterion_cycle(recovered_B, real_B)


       # Total generator loss
       loss_G = loss_GAN_A_to_B + loss_GAN_B_to_A + 10 * (loss_cycle_A + loss_cycle_B)
       loss_G.backward()
       optimizer_G.step()


       # Train Discriminators (minimize adversarial loss)
       optimizer_D_A.zero_grad()
       loss_D_A_real = criterion_GAN(model.discriminator_A(real_A), valid)
       loss_D_A_fake = criterion_GAN(model.discriminator_A(fake_A.detach()), fake_valid)
       loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2


       optimizer_D_B.zero_grad()
       loss_D_B_real = criterion_GAN(model.discriminator_B(real_B), valid)
       loss_D_B_fake = criterion_GAN(model.discriminator_B(fake_B.detach()), fake_valid)
       loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2


       # Backprop for discriminators
       loss_D_A.backward()
       loss_D_B.backward()


       optimizer_D_A.step()
       optimizer_D_B.step()


       # Print losses
       if i % 10 == 0:
           print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(young_loader)}] \
               Loss_G: {loss_G.item():.4f}, Loss_D_A: {loss_D_A.item():.4f}, Loss_D_B: {loss_D_B.item():.4f}")


   # Save checkpoints periodically
   if epoch % 10 == 0:
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_G_state_dict': optimizer_G.state_dict(),
           'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
           'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
       }, f"checkpoints/cyclegan_epoch_{epoch}.pth")


print("Training complete.")
