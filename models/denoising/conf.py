import cv2
import torch
import numpy as np

class Config:
    def __init__(self):
        self.device = 'cuda'
        self.lr = 0.001
        self.iterations = 300
        self.illu_factor = 1
        self.reflect_factor = 2
        self.noise_factor = 1000
        self.reffac = 2
        self.gamma = 3
        
        self.g_kernel_size = 5
        self.g_padding = 2
        self.sigma = 3
        self.kx = cv2.getGaussianKernel(self.g_kernel_size, self.sigma)
        self.ky = cv2.getGaussianKernel(self.g_kernel_size, self.sigma)
        self.gaussian_kernel = np.multiply(self.kx, np.transpose(self.ky))
        self.gaussian_kernel = torch.FloatTensor(self.gaussian_kernel).unsqueeze(0).unsqueeze(0).to(self.device)

conf = Config()