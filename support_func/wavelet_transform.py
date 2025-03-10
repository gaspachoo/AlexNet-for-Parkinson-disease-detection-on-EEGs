import numpy as np
import torch
import torch.nn.functional as F
from kymatio import Scattering1D
import matplotlib.pyplot as plt
import numpy

class WaveletScatteringTransform:
    def __init__(self, T, J=6, Q=3, out_shape=(227, 227, 3)):
        self.T = T
        self.J = J
        self.Q = Q
        self.out_shape = out_shape

        self.scattering = Scattering1D(J=self.J, shape=(self.T,), Q=self.Q, frontend='torch')

    def __call__(self, sample):
        labels = sample['labels']
        eeg_data = sample['eeg']

        T, num_channels = eeg_data.shape
        if T != self.T:
            raise ValueError(f"Signal length T={T} does not match the expected length T={self.T}.")

        eeg_tensor = torch.from_numpy(eeg_data)

        scattering_maps = []

        for ch in range(num_channels):
            signal_1d = eeg_tensor[:, ch].unsqueeze(0)
            Sx = self.scattering(signal_1d)
            
            #if isinstance(Sx, np.ndarray):
            #    Sx = torch.from_numpy(Sx)
            #else:
            #    Sx = Sx.float()
            
            Sx = Sx.squeeze(0).reshape(-1)

            num_coeffs = Sx.shape[0]
            side = int(np.ceil(np.sqrt(num_coeffs)))
            pad_size = side * side - num_coeffs
            if pad_size > 0:
                zeros_ = torch.zeros(pad_size, device=Sx.device, dtype=Sx.dtype)
                Sx = torch.cat([Sx, zeros_], dim=0)

            map_2d = Sx.reshape(side, side)
            scattering_maps.append(map_2d.unsqueeze(0))

        scattering_3d = torch.cat(scattering_maps, dim=0)

        c_per_group = num_channels // 3
        rgb_maps = []

        start_idx = 0
        for i in range(3):
            end_idx = min(start_idx + c_per_group, num_channels)
            group = scattering_3d[start_idx:end_idx]

            if group.shape[0] == 0:
                mean_map = torch.zeros(
                    (1, scattering_3d.shape[1], scattering_3d.shape[2]),
                    device=scattering_3d.device,
                    dtype=scattering_3d.dtype
                )
            else:
                mean_map = group.mean(dim=0, keepdim=True)

            rgb_maps.append(mean_map)
            start_idx += c_per_group

        rgb_tensor = torch.cat(rgb_maps, dim=0)

        rgb_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=self.out_shape[:2],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # ✅ Normalisation finale pour maximiser le contraste
        rgb_tensor = (rgb_tensor - rgb_tensor.min()) / (rgb_tensor.max() - rgb_tensor.min() + 1e-8)

        return {'data':rgb_tensor.float(),'labels':labels}

class WaveletScatteringTransform_1D:
    def __init__(self, T, J=6, Q=3, out_shape=(227, 227, 3)):
        self.T = T
        self.J = J
        self.Q = Q
        self.out_shape = out_shape

        self.scattering = Scattering1D(J=self.J, shape=(self.T,), Q=self.Q, frontend='torch')


    def __call__(self, sample):
        label = sample['label']
        eeg_data = sample['eeg']  # Expected shape: (T, num_channels)

        T, num_channels = eeg_data.shape
        if T != self.T:
            raise ValueError(f"Signal length T={T} does not match the expected length T={self.T}.")
        
        
        
        # ✅ Normalize EEG data at the beginning
        eeg_data = eeg_data / np.max(np.abs(eeg_data))  # Normalize signal between -1 and 1

        eeg_tensor = torch.from_numpy(eeg_data)

        # Extract only the CPz channel (already the only one in EEGDatasetIowa_1D)
        signal_1d = eeg_tensor[:, 0].unsqueeze(0)
        Sx = self.scattering(signal_1d)

        # Reshape scattering coefficients
        Sx = Sx.squeeze(0).reshape(-1)

        num_coeffs = Sx.shape[0]
        side = int(np.ceil(np.sqrt(num_coeffs)))
        pad_size = side * side - num_coeffs
        if pad_size > 0:
            zeros_ = torch.zeros(pad_size, device=Sx.device, dtype=Sx.dtype)
            Sx = torch.cat([Sx, zeros_], dim=0)

        map_2d = Sx.reshape(side, side).unsqueeze(0)  # 1-channel only (CPz)

        # Resize the image to match out_shape
        rgb_tensor = F.interpolate(
            map_2d.unsqueeze(0),
            size=self.out_shape[:2],  # Keep height and width
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return {'data': rgb_tensor.float(), 'label': label}

class WaveletScatteringTransformTFR_RGB:
    def __init__(self, T, J=6, Q=1, frontend='torch'):
        self.T = T
        self.J = J
        self.Q = Q
        self.scattering = Scattering1D(
            J=self.J, shape=(self.T,), Q=self.Q, frontend=frontend
        )

    def __call__(self, sample):
        label = sample['label']
        # sample['eeg'] assumed shape (T,) or (T,1). We'll assume (T,).
        eeg_data = sample['eeg'].astype(np.float32)

        # Turn into Torch tensor: [1, T]
        x = torch.from_numpy(eeg_data).unsqueeze(0)
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
        
        colormap = plt.cm.viridis  # Get the colormap
        tensor_rgb = colormap(grayscale_tensor.numpy())[...,:3]  # Get RGB channels (ignore alpha)
        
        # Convert RGB NumPy array to PyTorch tensor
        tensor_rgb_torch = torch.tensor(tensor_rgb, dtype=torch.float32)  # Shape: (227, 227, 3)

        # If you need it in (3, 227, 227) format (channel-first for deep learning models)
        tensor_rgb_torch = tensor_rgb_torch.permute(2, 0, 1)  # Shape: (3, 227, 227)
        
        # For direct TFR plotting, we might just return it as is:
        return {
            'tfr': tensor_rgb_torch,   # shape: [num_scales, T']
            'label': label
        }

class WaveletScatteringTransformTFR:
    def __init__(self, T, J=6, Q=1, frontend='torch'):
        self.T = T
        self.J = J
        self.Q = Q
        self.scattering = Scattering1D(
            J=self.J, shape=(self.T,), Q=self.Q, frontend=frontend
        )

    def __call__(self, sample):
        label = sample['label']
        # sample['eeg'] assumed shape (T,) or (T,1). We'll assume (T,).
        eeg_data = sample['eeg'].astype(np.float32)

        # Turn into Torch tensor: [1, T]
        x = torch.from_numpy(eeg_data).unsqueeze(0)
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
        
        # For direct TFR plotting, we might just return it as is:
        return {
            'tfr': grayscale_tensor,   # shape: [num_scales, T']
            'label': label
        }