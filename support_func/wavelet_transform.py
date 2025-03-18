import numpy as np
import torch
import torch.nn.functional as F
from kymatio import Scattering1D
import matplotlib.pyplot as plt

class WaveletScatteringTransform:
    def __init__(self, T, J=8, Q=16, frontend='torch'):
        self.T = T
        self.J = J
        self.Q = Q
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scattering = Scattering1D(J=self.J, shape=(self.T,), Q=self.Q, frontend=frontend).to(self.device)

    def __call__(self, sample):
        label = sample['label']
        # sample['eeg'] assumed shape (T,).
        eeg_data = sample['eeg'].astype(np.float32)
        

        #eeg_data = eeg_data / np.max(np.abs(eeg_data))
        scaling_factor = 1e5  # Adjust based on min/max values
        #eeg_data *= scaling_factor

        # Turn into Torch tensor: [1, T]
        x = torch.from_numpy(eeg_data).unsqueeze(0).to(self.device)
        
        x = x.squeeze(-1)
        
        #print("DEBUG: x.shape =", x.shape,flush= True)
        #print("DEBUG: self.scattering.J =", self.scattering.J,flush=True)
        #print("DEBUG: self.scattering.shape =", self.scattering.shape,flush=True)
        
        # Forward pass => shape [1, C, T']
        Sx = self.scattering(x)
               
        # Retrieve meta to identify 1st-order channels
        meta = self.scattering.meta()
        order_array = meta['order']  # array of shape [C]
        idx_first_order = np.where(order_array == 1)[0]

        # Just keep first-order channels
        # Sx shape: [1, C, T'] => pick only the channels in idx_first_order
        S1 = Sx[:, idx_first_order, :].squeeze(0)  # => shape [num_scales, T']

        # At this point, S1 is 2D: (freq_scales, time_frames).
        
        # ✅ Resize to match output shape (227x227)
        grayscale_tensor = F.interpolate(
            S1.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
            size=(227, 227),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # Remove batch & channel dims
        
        grayscale_tensor = (grayscale_tensor - grayscale_tensor.min()) / (grayscale_tensor.max() - grayscale_tensor.min() + 1e-8)


        
        colormap = plt.cm.viridis  # Get the colormap
        tensor_rgb = colormap(grayscale_tensor.cpu().numpy())[..., :3]  # Convert to NumPy and extract RGB channels
        
        # Convert RGB NumPy array to PyTorch tensor
        tensor_rgb_torch = torch.tensor(tensor_rgb, dtype=torch.float32)    # Shape: (227, 227, 3)

        # If you need it in (3, 227, 227) format (channel-first for deep learning models)
        tensor_rgb_torch = tensor_rgb_torch.permute(2, 0, 1)  # Shape: (3, 227, 227)
        
        return {
            'image': tensor_rgb_torch,   # shape: [num_scales, T']
            'label': label
        }
