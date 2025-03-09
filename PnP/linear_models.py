import numpy as np
from scipy.signal import convolve2d

import torch
import torch.nn.functional as F
import pywt

from utils import pad, unpad
from scipy.ndimage import gaussian_filter

def imfilter(x, k, transposed=False, n_channel=3):
    '''
    Equivalent (verified) to scipy ndimage.convolve with mode='wrap'.
    x: image, NxcxHxW
    k: kernel, hxw
    '''
    n_channel = x.shape[1]
    k = k.repeat(n_channel, 1, 1, 1)
    k = k.flip(-1).flip(-2) # flip kernel for convolution and not correlation !!!
    ph = (k.shape[-2] - 1)//2
    pw = (k.shape[-1] - 1)//2
    if not transposed:
        x = pad(x, p=ph)
        x = F.conv2d(x, k, groups=x.shape[1])
    else :
        x = F.conv_transpose2d(x, k, groups=x.shape[1])
        x = unpad(x, ph, pw)
    return x

def blur_tensor(img, ker, orig_dim):
    # Apply convolution channel-wise if input has channels
    # ker_size = ker.size(-1)
    if orig_dim == 3 or orig_dim == 4:
        for i in range(img.shape[1]):
            if i==0:
                out = F.conv2d(img[:, i:i+1], ker, padding=0)
            else:
                out = torch.cat((out, F.conv2d(img[:, i:i+1], ker, padding=0)), dim=1)
    else:
        out = F.conv2d(img, ker, padding=0)
    return out

def downsample(img, block_size):
    """
    Downsamples image by taking every block_size-th pixel
    
    Args:
        img (torch.Tensor): Input tensor of shape (H,W) or (C,H,W)
        block_size (int): Downsampling factor
    
    Returns:
        torch.Tensor: Downsampled tensor preserving input dimensions
    """

    if img.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {img.dim()}D")
    
    return img[:, :, ::block_size, ::block_size]

def downsample_transpose(img, block_size):
    """
    Transpose operation of downsampling - inserts zeros between pixels
    
    Args:
        img (torch.Tensor): Input tensor of shape (H,W) or (C,H,W)
        block_size (int): Upsampling factor
    
    Returns:
        torch.Tensor: Upsampled tensor with zeros, preserving input dimensions
    """

    if img.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {img.dim()}D")
    
    
    B, C, H, W = img.shape
    # Create output tensor
    out = torch.zeros((B, C, H * block_size, W * block_size), 
                        dtype=img.dtype, 
                        device=img.device)
    # Fill values
    out[:, :, ::block_size, ::block_size] = img
    
    return out

class Operators:
    def __init__(self):
        pass

    def _verify_operator(self):
        operator = self.operator
        # Convert operator to torch tensor if it isn't already
        if not isinstance(operator, torch.Tensor):
            operator = torch.tensor(operator, device=self.device, dtype=torch.float32)

        # Add dimensions for batch and channel if needed
        if operator.dim() == 2:
            operator = operator.unsqueeze(0).unsqueeze(0)
        elif operator.dim() == 3:
            operator = operator.unsqueeze(0)
        self.operator = operator

    def _verify_input(self, img):
        # Convert input to tensor if it isn't already
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, device=self.device, dtype=torch.float32)

        # Handle different input dimensions
        orig_dim = img.dim()
        if orig_dim == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif orig_dim == 3:
            img = img.unsqueeze(0)  # Add only batch dim
        return img, orig_dim

    def _pad_tensor(self, x, pad_size):
        """Helper function to pad tensor with circular padding"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='circular')

class BlurOperator(Operators):
    def __init__(self, operator, mode, device):
        """
        Initialize the image operator
        Args:
            operator: kernel for convolution or mask for inpainting
            mode: "blur", "supr" (super-resolution), or "inpnt" (inpainting)
            size: kernel size for blur mode
            res: resolution factor for super-resolution mode
        """
        self.mode = mode
        self.device = device

        # self.operator = operator
        # self.size = operator.shape[0]
        self.sigma = operator['sigma']
        self.radius = operator['radius']

        # self._verify_operator()

    def A(self, img: torch.tensor):
        """Forward operator"""
        out = torch.tensor(gaussian_filter(input=img.clone().detach().to('cpu'), sigma=self.sigma, mode='mirror', radius = self.radius)).to(dtype=img.dtype, device=img.device)
        # img, orig_dim = self._verify_input(img)
        # pad_size = self.size // 2
        # img = self._pad_tensor(img, pad_size)

        # # Apply convolution channel-wise if input has channels
        # out = blur_tensor(img, self.operator, orig_dim)

        # # Restore original dimensions
        # if orig_dim == 2:
        #     out = out.squeeze(0).squeeze(0)
        # elif orig_dim == 3:
        #     out = out.squeeze(0)

        return out

    def At(self, img: torch.tensor):
        """Adjoint operator"""
        out = torch.tensor(gaussian_filter(input=img.clone().detach().to('cpu'), sigma=self.sigma, mode='mirror', radius = self.radius)).to(dtype=img.dtype, device=img.device)
        # out = torch.tensor(gaussian_filter(input=img, sigma=self.sigma, mode='mirror', radius = self.radius)).to(dtype=img.dtype, device=img.device)
        # Convert input to tensor if it isn't already        
        # img, orig_dim = self._verify_input(img)

        # pad_size = self.size // 2
        # img = self._pad_tensor(img, pad_size)
        
        # # Flip kernel for adjoint operation
        # flipped_operator = torch.flip(self.operator, dims=[-1, -2])
        
        # out = blur_tensor(img, flipped_operator, orig_dim)
        
        # # Restore original dimensions
        # if orig_dim == 2:
        #     out = out.squeeze(0).squeeze(0)
        # elif orig_dim == 3:
        #     out = out.squeeze(0)
            
        return out

class Motion_BlurOperator(Operators):
    def __init__(self, operator, mode, device):
        """
        Initialize the image operator
        Args:
            operator: kernel for convolution or mask for inpainting
            mode: "Mo-blur"
            size: kernel size for blur mode
            res: resolution factor for super-resolution mode
        """
        self.mode = mode
        self.device = device

        self.operator = operator
        self.size = operator.shape[0]

        self._verify_operator()

    def A(self, img: torch.tensor):
        """Forward operator"""
        img, orig_dim = self._verify_input(img)

        # Apply convolution channel-wise if input has channels
        out = imfilter(img, self.operator, transposed=False, n_channel=img.shape[1])

        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)

        return out

    def At(self, img: torch.tensor):
        """Adjoint operator"""
        # Convert input to tensor if it isn't already        
        img, orig_dim = self._verify_input(img)

        out = imfilter(img, self.operator, transposed=True, n_channel=img.shape[1])

        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)

        return out

class SupResOperator(Operators):
    def __init__(self, operator, mode, device):
        """
        Initialize the image operator
        Args:
            operator: kernel for convolution or mask for inpainting
            mode: "blur", "supr" (super-resolution), or "inpnt" (inpainting)
            size: kernel size for blur mode
            res: resolution factor for super-resolution mode
        """
        self.mode = mode
        self.device = device
        
        self.operator = operator["operator"]
        self.res = operator["res"]
        self.size = operator["operator"].shape[-1]

        self._verify_operator()

    def A(self, img: torch.tensor):
        """Forward operator"""
        img, orig_dim = self._verify_input(img)
        
        # Convolution blurring
        pad_size = self.size // 2
        img = self._pad_tensor(img, pad_size)
        conv_out = blur_tensor(img, self.operator)
        
        # Downsample
        out = downsample(conv_out, self.res)
        
        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)

        return out

    def At(self, img: torch.tensor):
        """Adjoint operator"""
        # Convert input to tensor if it isn't already        
        img, orig_dim = self._verify_input(img)

        # Transpose of upsample is average pooling
        # Transpose of downsample is nearest neighbor upsampling
        down_t = downsample_transpose(down_t, self.res)
        pad_size = self.size // 2
        down_t = self._pad_tensor(down_t, pad_size)

        # Flip kernel for adjoint operation
        flipped_operator = torch.flip(self.operator, dims=[-1, -2])

        # Channel-wise operations for convolution
        out = blur_tensor(down_t, flipped_operator)

        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)

        return out

class InpntOperator(Operators):
    def __init__(self, operator, mode, device):
        """
        Initialize the image operator
        Args:
            operator: kernel for convolution or mask for inpainting
            mode: "blur", "supr" (super-resolution), or "inpnt" (inpainting)
            size: kernel size for blur mode
            res: resolution factor for super-resolution mode
        """
        self.mode = mode
        self.device = device
        
        self.operator = operator
        
        self._verify_operator()

    def A(self, img: torch.tensor):
        """Forward operator"""
        img, orig_dim = self._verify_input(img)

        out = img * self.operator.expand_as(img)
        
        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)
            
        return out

    def At(self, img: torch.tensor):
        """Adjoint operator"""
        # Convert input to tensor if it isn't already        
        img, orig_dim = self._verify_input(img)
    
        out = img * self.operator.expand_as(img)
        
        # Restore original dimensions
        if orig_dim == 2:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            out = out.squeeze(0)
            
        return out


class linear:
    def __init__(self, mode, oper, device, lvl = 0, wave='haar'):
        """
        Initializes the operator based on the specified inverse problem.
        Parameters:\n
        mode (str): The mode of the operator. Can be \n
                    - 'blur' - Blurring, 
                    - 'Mo-blur' - Motion Blurring,
                    - 'supR' - Super Resolution, 
                    - 'inpnt'- Inpainting.
        oper (any): Variable for specific operation to be performed by the operator.\n
                    - For 'blur': oper(Dict) {'sigma': -, 'radius': -}.
                    - For 'Mo-blur': oper(Tensor) 2D bluring kernel shape: (k,k).
                    - For 'supR': oper(Dict) {"operator": blur kernel shape(1,1,k,k), "res": Downsampling factor}.
                    - For 'inpnt': oper(Tensor) Mask of shape same image.
        Raises:
        ValueError: If the specified mode is not implemented.
        """
        if mode == 'blur':
            self.oper = BlurOperator(mode=mode, operator=oper, device=device)
        if mode == 'Mo-blur':
            self.oper = Motion_BlurOperator(mode=mode, operator=oper, device=device)
        elif mode == 'supR':
            self.oper = SupResOperator(mode=mode, operator=oper, device=device)
        elif mode == 'inpnt':
            self.oper = InpntOperator(mode=mode, operator=oper, device=device)
        else:
            raise ValueError(f'Operator {mode} has not been implemented.')

    def A(self, x: torch.tensor):
        return self.oper.A(x)

    def At(self, x: torch.tensor):
        return self.oper.At(x)
    
    def leastsq_A(self, x: torch.tensor, gamma: float=1.0):
        return self.oper.leastsq_A(x, gamma)
    
    def function_value(self, x: torch.tensor, b: torch.tensor):
        return 1/2 * torch.norm(self.A(x) - b, p='fro')**2