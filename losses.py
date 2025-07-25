import torch
import torch.nn as nn


class GANLoss(nn.Module):
   """Define the GAN loss function"""
   def __init__(self, loss_type="mse"):
       super(GANLoss, self).__init__()
       if loss_type == "mse":
           self.loss = nn.MSELoss()
       elif loss_type == "bce":
           self.loss = nn.BCELoss()
       else:
           raise ValueError("Invalid loss type. Choose 'mse' or 'bce'.")


   def forward(self, prediction, target):
       return self.loss(prediction, target)


class CycleConsistencyLoss(nn.Module):
   """Define the cycle consistency loss"""
   def __init__(self, lambda_cycle=10):
       super(CycleConsistencyLoss, self).__init__()
       self.lambda_cycle = lambda_cycle
       self.loss = nn.L1Loss()


   def forward(self, real_image, reconstructed_image):
       return self.lambda_cycle * self.loss(real_image, reconstructed_image)


class IdentityLoss(nn.Module):
   """Define the identity loss"""
   def __init__(self, lambda_identity=5):
       super(IdentityLoss, self).__init__()
       self.lambda_identity = lambda_identity
       self.loss = nn.L1Loss()


   def forward(self, real_image, same_image):
       return self.lambda_identity * self.loss(real_image, same_image)


