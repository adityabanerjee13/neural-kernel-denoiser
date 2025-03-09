import numpy as np
import os
import math
from datetime import datetime
import torch
import torch.nn as nn
from einops import rearrange
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import pickle
import h5py






def gradient_transform(x):
    outsize = (x.size(0), x.size(1), x.size(2)-1, x.size(3)-1)
    out = torch.zeros(outsize)
    dx = torch.tensor([[[[-1,1]]]], dtype=x.dtype, device = x.device)
    dy = torch.tensor([[[[-1],[1]]]], dtype=x.dtype, device = x.device)

    for i in range(x.size(1)):
        temp = torch.functional.F.conv2d(x[:, i:i+1, :, :], dx)
        out[:, i:i+1, :, :] = torch.functional.F.conv2d(temp, dy)

    return out

def sRGB(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x**(1 / 2.4) - 0.055)

def extract_motion_blur_kernels(random=False, index=None):
    kernel_path = './kernels/Levin09.mat'
    # Extract actual kernel data from HDF5 object references
    with h5py.File(kernel_path, 'r') as file:
        data = file['kernels']
        # Extract data from the HDF5 dataset
        kernels = np.array(data)
        kernel_list = [np.array(file[ref]) for ref in kernels.flatten()]

    if random:
        index = np.random.choice(len(kernel_list), 1)[0]

    if index is not None:
        kernel = kernel_list[index]
    else:
        kernel = kernel_list[0]

    return kernel


def date_time_stamp(format = "%m%d%H%M%S"):
    return datetime.now().strftime(format)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def linear_img_fwd(img, parameters: nn.Linear):
    out_channel, in_channel = parameters.weight.shape
    parameters_linear = (parameters.weight).reshape(out_channel,in_channel, 1, 1)
    parameters_bias = parameters.bias
    return torch.functional.F.conv2d(
        img,parameters_linear,
        bias=parameters_bias,
    )

def norm_img_fwd(img, parameters: nn.LayerNorm):
    parameters_linear = parameters.weight
    parameters_bias = parameters.bias
    c, h, w = img.shape[-3:]
    inp = img2channel_list(img)
    out = torch.functional.F.layer_norm(inp,(c,),parameters_linear,bias=parameters_bias)
    return channel_list2img(out, h, w)

def blk_img_fwd(x, blk):
    for i in blk:
        if isinstance(i, nn.Linear):
            x = linear_img_fwd(x, i)
        elif isinstance(i, nn.LayerNorm):
            x = norm_img_fwd(x, i)
        elif isinstance(i, nn.Sequential):
            x = blk_img_fwd(x, i)
        else:
            x = i(x)
    return x

def unpad(img, hp, wp):
    if len(img.shape)==3:
        return img[:, hp:-hp, wp:-wp]
    elif len(img.shape)==4:
        return img[:, :, hp:-hp, wp:-wp]
    else:
        raise ValueError(f'image shape {img.shape} is invalid')

def pad(img, p, mode='circular'):
    return torch.functional.F.pad(img, (p,p,p,p), mode=mode)

def img2channel_list(img):
    if len(img.shape)==3:
        c, h, w = img.shape
        return img.permute(1,2,0).reshape(-1,c)
    elif len(img.shape)==4:
        b, c, _, _ = img.shape
        return img.permute(0,2,3,1).reshape(b, -1, c)

def channel_list2img(channel_list, h, w):
    if len(channel_list.shape)==2:
        n, c = channel_list.shape
        return channel_list.reshape(h, w ,c).permute(2,0,1)
    elif len(channel_list.shape)==3:
        b, n, c = channel_list.shape
        return channel_list.reshape(b, h, w ,c).permute(0,3,1,2)

def extract_all_patches(img, patch_rad):
    # patch_rad = patch_size//2
    C, H, W = img.shape
    # Randomly select points
    y_coords = torch.arange(patch_rad, H - patch_rad, 1)
    x_coords = torch.arange(patch_rad, W - patch_rad, 1)
    # Extract patches
    img_patches=[]
    for i in y_coords:
        img_patches += list_of_pixels2patches(img, torch.full_like(x_coords, i), x_coords, patch_rad)
    return torch.stack(img_patches)

def list_of_pixels2patches(img, y_coords, x_coords, patch_rad):
    # Extract patches
    img_patches = []
    for y, x in zip(y_coords, x_coords):
        patch = img[:, y - patch_rad:y + patch_rad + 1, x - patch_rad:x + patch_rad + 1]
        img_patches.append(patch)
    return img_patches

def load_checkpoint(checkpoint_path, architecture, device, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    # Load the checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = architecture(
        patch_rad=state['args'].patch_rad,
        window_rad=state['args'].window_rad,
        in_channel=state['args'].in_channel,
        latent_dim=state['args'].latent_dim,
        latent_channel_dim=state['args'].latent_channel_dim,
        device=device,
        blind=state['args'].blind,
    ).to(device)

    model.load_state_dict(state['model_state_dict'])
    print(f"Loaded model state from checkpoint: {checkpoint_path}")

    # Optionally load optimizer state
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Loaded optimizer state.")

    # Optionally load scheduler state
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
        print("Loaded scheduler state.")

    # Return other metadata like epoch, step, loss, etc.
    return model, optimizer, scheduler, {
        'step': state.get('step', None),
        'epoch': state.get('epoch', None),
        'loss': state.get('loss', None),
        'psnr': state.get('psnr', None),
        'ssim': state.get('ssim', None),
        'best_psnr': state.get('best_psnr', None),
        'best_ssim': state.get('best_ssim', None),
    }

def save_checkpoint(out_folder, state, step, is_best=False, _state=None):
    """Helper function to save checkpoints"""
    if state == None:
        _state = f'_{state['args'].patch_rad}{state['args'].window_rad}{state['args'].in_channel}{state['args'].latent_dim}{state['args'].latent_channel_dim}{int(state['args'].blind)}{state['args'].sigma_min}_{state['args'].sigma_max}'
    else:
        pass
    if is_best:
        checkpoint_path = os.path.join(out_folder, f'model{_state}_best.pth')
    else:
        checkpoint_path = os.path.join(out_folder, f'model{_state}_step_{step}.pth')
    
    torch.save(state, checkpoint_path)

def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)],dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)],dim=-1)
    # grid_w_emb -> (Number of patch tokens, pos_emb_dim // 2)

    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)
    # pos_emb -> (Number of patch tokens, pos_emb_dim)

    return rearrange(pos_emb, '(h w) d -> d h w', h=grid_size_h, w=grid_size_w)

def calculate_psnr(batch1, batch2):
    psnr_scores = []
    if len(batch1.shape)==4 and len(batch2.shape)==4:
        batch1_np = batch1.permute(0, 2, 3, 1).detach().cpu().numpy()  
        batch2_np = batch2.permute(0, 2, 3, 1).detach().cpu().numpy()  

        # Compute PSNR for each image in the batch
        for i in range(batch1_np.shape[0]):  # Loop over the batch
            img1 = batch1_np[i]
            img2 = batch2_np[i]

            # Compute PSNR for each channel and average the scores
            psnr_val = psnr(img1, img2, data_range=1)
            psnr_scores.append(psnr_val)

    elif len(batch1.shape)==3 and len(batch2.shape)==3:
        batch1_np = batch1.permute(1, 2, 0).detach().cpu().numpy()  
        batch2_np = batch2.permute(1, 2, 0).detach().cpu().numpy()  
        psnr_val = psnr(batch1_np, batch2_np, data_range=1)
        psnr_scores.append(psnr_val)
    
    else:
        raise ValueError(f'error in shape of input 1:{batch1.shape}, 2:{batch2.shape}')

    # Return mean PSNR scores.
    return np.mean(psnr_scores)

def img_ssim(img1, img2, channels):
    # Compute SSIM for each channel and average the scores
    per_channel_ssim = [
        ssim(img1[:, :, c], img2[:, :, c], data_range=img2[:, :, c].max() - img2[:, :, c].min())
        for c in range(channels)
    ]
    return np.mean(per_channel_ssim)  # Average across channels

def calculate_ssim(batch1, batch2):
    
    ssim_scores = []
    if len(batch1.shape)==4 and len(batch1.shape)==4:
        channels = batch1.size(1)
        batch1_np = batch1.permute(0, 2, 3, 1).detach().cpu().numpy()  
        batch2_np = batch2.permute(0, 2, 3, 1).detach().cpu().numpy()  

        # Compute SSIM for each image in the batch
        for i in range(batch1_np.shape[0]):  # Loop over the batch
            img1 = batch1_np[i]
            img2 = batch2_np[i]
            mean_ssim = img_ssim(img1, img2, channels)
            ssim_scores.append(mean_ssim)
    elif len(batch1.shape)==3 and len(batch1.shape)==3:
        channels = batch1.size(0)
        batch1_np = batch1.permute(1, 2, 0).detach().cpu().numpy()  
        batch2_np = batch2.permute(1, 2, 0).detach().cpu().numpy() 
        mean_ssim = img_ssim(batch1_np, batch2_np, channels)
        ssim_scores.append(mean_ssim)
    else:
        raise ValueError(f'error in shape of input 1:{batch1.shape}, 2:{batch2.shape}')

    # Return mean SSIM scores.
    return np.mean(ssim_scores)

def show_tensor_image(image_tensors : list, grid : tuple, titles=None, save=False, save_path=None, vmin=0, vmax=1):
    x, y = grid
    _, axs = plt.subplots(x, y, figsize=(5*y, 5*x))
    # Convert tensors to numpy arrays for plotting
    if not isinstance(image_tensors, list):
        image_tensors = [image_tensors]
    if titles is not None:
        if not isinstance(titles, list):
            titles = [titles]
        assert len(image_tensors) == len(titles), 'Number of images and titles must be the same.'

    for i in range(x):
        for j in range(y):
            index = i*y + j
            if index >= len(image_tensors):
                break
            np_image = image_tensors[index].detach().cpu().numpy()
            if len(np_image.shape)==4:
                np_image = np_image.squeeze(0).transpose(1, 2, 0)
            else:
                np_image = np_image.transpose(1, 2, 0)
            if y>1 and x>1:
                axs[i,j].imshow(np_image, cmap='gray', vmin=vmin, vmax=vmax)
                if titles:
                    axs[i,j].set_title(titles[index])
                axs[i,j].axis('off')

            elif y>1:
                axs[j].imshow(np_image, cmap='gray', vmin=vmin, vmax=vmax)
                if titles:
                    axs[j].set_title(titles[index])
                axs[j].axis('off')

            elif x>1:
                axs[i].imshow(np_image, cmap='gray', vmin=vmin, vmax=vmax)
                if titles:
                    axs[i].set_title(titles[index])
                axs[i].axis('off')
            else:
                plt.imshow(np_image, cmap='gray', vmin=vmin, vmax=vmax)
                if titles:
                    plt.title(titles[index])
                plt.axis('off')

    # plt.tight_layout()
    if save: plt.savefig(save_path)
    else: plt.show()
    plt.close()

def img2Dshift(v, dx, dy):
    typ = (1 if dx > 0 else 0) * 2 + (1 if dy > 0 else 0)
    if len(v.shape) == 4:
        batch, c, row, col = v.shape
        t = torch.zeros((batch, c, row, col), device=v.device, dtype=v.dtype)

        if typ == 0:  # dx <= 0, dy <= 0
            t[:,:,-dx:, -dy:] = v[:,:,:row + dx, :col + dy]
        elif typ == 1:  # dx <= 0, dy > 0
            t[:,:,-dx:, :col - dy] = v[:,:,:row + dx, dy:]
        elif typ == 2:  # dx > 0, dy <= 0
            t[:,:,:row - dx, -dy:] = v[:,:,dx:, :col + dy]
        elif typ == 3:  # dx > 0, dy > 0
            t[:,:,:row - dx, :col - dy] = v[:,:,dx:, dy:]
    elif len(v.shape) == 5:
        batch, n, c, row, col = v.shape
        t = torch.zeros((batch, n, c, row, col), device=v.device, dtype=v.dtype)

        if typ == 0:  # dx <= 0, dy <= 0
            t[:,:,:,-dx:, -dy:] = v[:,:,:,:row + dx, :col + dy]
        elif typ == 1:  # dx <= 0, dy > 0
            t[:,:,:,-dx:, :col - dy] = v[:,:,:,:row + dx, dy:]
        elif typ == 2:  # dx > 0, dy <= 0
            t[:,:,:,:row - dx, -dy:] = v[:,:,:,dx:, :col + dy]
        elif typ == 3:  # dx > 0, dy > 0
            t[:,:,:,:row - dx, :col - dy] = v[:,:,:,dx:, dy:]
    else:
        batch, row, col = v.shape
        t = torch.zeros((batch, row, col), device=v.device, dtype=v.dtype)

        if typ == 0:  # dx <= 0, dy <= 0
            t[:,-dx:, -dy:] = v[:,:row + dx, :col + dy]
        elif typ == 1:  # dx <= 0, dy > 0
            t[:,-dx:, :col - dy] = v[:,:row + dx, dy:]
        elif typ == 2:  # dx > 0, dy <= 0
            t[:,:row - dx, -dy:] = v[:,dx:, :col + dy]
        elif typ == 3:  # dx > 0, dy > 0
            t[:,:row - dx, :col - dy] = v[:,dx:, dy:]
    return t

def img_sq_diff(v, dx, dy):
    t = img2Dshift(v, dx, dy)
    diff = (v - t)**2
    return diff

def separable_triangle(dx, dy, Ns, height, width):

    # Calculate the value from dx, dy, and Ns
    value = max(0, 1 - (abs(dx) / (Ns + 1))) * max(0, 1 - (abs(dy) / (Ns + 1)))
    # value = 1

    # Build the 2D array Hat with all values equal to `value`
    Hat = torch.full((height, width), value, dtype=torch.float32)

    # Adjust pixel values to zero where the destination pixel is outside the image
    if dx >= 0:
        Hat[height - dx:, :] = 0
    else:
        Hat[:abs(dx), :] = 0

    if dy >= 0:
        Hat[:, width - dy:] = 0
    else:
        Hat[:, :abs(dy)] = 0

    return Hat

def Box(dx, dy, Ns, height, width):

    # Calculate the value from dx, dy, and Ns
    value = 1

    # Build the 2D array Hat with all values equal to `value`
    Hat = torch.full((height, width), value, dtype=torch.float32)

    # Adjust pixel values to zero where the destination pixel is outside the image
    if dx >= 0:
        Hat[height - dx:, :] = 0
    else:
        Hat[:abs(dx), :] = 0

    if dy >= 0:
        Hat[:, width - dy:] = 0
    else:
        Hat[:, :abs(dy)] = 0

    return Hat
