import torch
from deepinv.models import BM3D

from models.NKD_linear_layer import NKD_latent_layers, NLM
from denoisers.NLM_torch import NLM_tensor
from denoisers.TV import TV
# from utils import load_checkpoint, save_checkpoint



def load_denoiser(architecture: torch.nn.Module, checkpoint: str, device: str) -> torch.nn.Module:
    """
    Loads a denoiser model from a checkpoint file.

    Args:
        architecture (any): The architecture class of the denoiser model.
        checkpoint (str): Path to the checkpoint file containing the model state.
        device (str): The device on which to load the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded denoiser model with the state from the checkpoint.

    """

    states = torch.load(checkpoint, weights_only=False)
    denoiser = architecture(
        patch_rad=states['args'].patch_rad, 
        window_rad=states['args'].window_rad,
        in_channel=states['args'].in_channel,
        latent_dim=states['args'].latent_dim,
        latent_channel_dim=states['args'].latent_channel_dim,
        device=device,
        blind=states['args'].blind,
    )
    denoiser.load_state_dict(states['model_state_dict'])
    print('Denoiser Training Information:')
    print('patch_rad=',states['args'].patch_rad)
    print('window_rad=',states['args'].window_rad)
    print('in_channel=',states['args'].in_channel)
    print('latent_dim=',states['args'].latent_dim)
    print('latent_channel_dim=',states['args'].latent_channel_dim)
    print('blind=',states['args'].blind)

    return denoiser


class NKD_regularizer:
    def __init__(self, architecture : torch.nn.Module, checkpoint : str, device : str, window_rad : int = 5):
        """
        Initializes the class with the given architecture, checkpoint, and device.
        Args:
            architecture (class): The architecture type for the denoiser.
            checkpoint (str): The path to the checkpoint file for loading the denoiser.
            device (str): The device to run the denoiser on (e.g., 'cpu' or 'cuda').
        Attributes:
            denoiser: The loaded denoiser model.
            device (str): The device to run the denoiser on.
        """
        self.denoiser = load_denoiser(architecture, checkpoint, device).to(device)
        self.denoiser.window_rad = window_rad

    def __call__(self, noisy_input, guide, sigma):
        with torch.no_grad():
            feat = self.denoiser.get_features(guide, sigma)
            out = NLM(noisy_input, feat, self.denoiser.window_rad, self.denoiser.sigma)
        return out
    


class NLM_regularizer:
    def __init__(self, patch_rad, window_rad, sigma):
        """
        Initializes the class with the given architecture, checkpoint, and device.
        Args:
            architecture (class): The architecture type for the denoiser.
            checkpoint (str): The path to the checkpoint file for loading the denoiser.
            device (str): The device to run the denoiser on (e.g., 'cpu' or 'cuda').
        Attributes:
            denoiser: The loaded denoiser model.
            device (str): The device to run the denoiser on.
        """
        self.window_rad = window_rad
        self.patch_rad = patch_rad
        self.sigma = sigma

    def __call__(self, noisy_input, guide, sigma):
        with torch.no_grad():
            out = NLM_tensor(noisy_input, guide, self.patch_rad, self.window_rad, self.sigma, False)
        return out
    


class BM3D_regularizer:
    def __init__(self):
        """
        Initializes the class with the given architecture, checkpoint, and device.
        Args:
            architecture (class): The architecture type for the denoiser.
            checkpoint (str): The path to the checkpoint file for loading the denoiser.
            device (str): The device to run the denoiser on (e.g., 'cpu' or 'cuda').
        Attributes:
            denoiser: The loaded denoiser model.
            device (str): The device to run the denoiser on.
        """
        self.denoiser = BM3D()

    def __call__(self, noisy_input, guide, sigma):
        device = noisy_input.device
        with torch.no_grad():
            out = self.denoiser(noisy_input.to('cpu'), sigma.item())
        return out.to(device)
    


class TV_regularizer(TV):
    pass
