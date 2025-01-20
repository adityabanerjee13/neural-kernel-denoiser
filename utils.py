import numpy as np
import os
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


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

def save_checkpoint(out_folder, state, step, is_best=False):
    """Helper function to save checkpoints"""
    _state = f'_{state['args'].patch_rad}{state['args'].window_rad}{state['args'].in_channel}{state['args'].latent_dim}{state['args'].latent_channel_dim}{int(state['args'].blind)}{state['args'].sigma_min}_{state['args'].sigma_max}'
    if is_best:
        checkpoint_path = os.path.join(out_folder, f'model{_state}_best.pth')
    else:
        checkpoint_path = os.path.join(out_folder, f'model{_state}_step_{step}.pth')
    
    torch.save(state, checkpoint_path)

def calculate_psnr(batch1, batch2):
    batch1_np = batch1.permute(0, 2, 3, 1).detach().cpu().numpy()  
    batch2_np = batch2.permute(0, 2, 3, 1).detach().cpu().numpy()  

    # Compute PSNR for each image in the batch
    psnr_scores = []
    for i in range(batch1_np.shape[0]):  # Loop over the batch
        img1 = batch1_np[i]
        img2 = batch2_np[i]

        # Compute PSNR for each channel and average the scores
        psnr_val = psnr(img1, img2, data_range=1)
        psnr_scores.append(psnr_val)

    # Return mean PSNR scores.
    return np.mean(psnr_scores)

def calculate_ssim(batch1, batch2):
    batch1_np = batch1.permute(0, 2, 3, 1).detach().cpu().numpy()  
    batch2_np = batch2.permute(0, 2, 3, 1).detach().cpu().numpy()  

    # Compute SSIM for each image in the batch
    ssim_scores = []
    for i in range(batch1_np.shape[0]):  # Loop over the batch
        img1 = batch1_np[i]
        img2 = batch2_np[i]
        
        # Compute SSIM for each channel and average the scores
        per_channel_ssim = [
            ssim(img1[:, :, c], img2[:, :, c], data_range=img2[:, :, c].max() - img2[:, :, c].min())
            for c in range(3)
        ]
        mean_ssim = np.mean(per_channel_ssim)  # Average across channels
        ssim_scores.append(mean_ssim)

    # Return mean SSIM scores.
    return np.mean(ssim_scores)

def show_tensor_image(image_tensors : list, grid : tuple, titles=None, save=False, save_path=None):
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
            np_image = image_tensors[index].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            if y>1 and x>1:
                axs[i,j].imshow(np_image)
                if titles:
                    axs[i,j].set_title(titles[index])
                axs[i,j].axis('off')

            elif y>1:
                axs[j].imshow(np_image)
                if titles:
                    axs[j].set_title(titles[index])
                axs[j].axis('off')

            elif x>1:
                axs[i].imshow(np_image)
                if titles:
                    axs[i].set_title(titles[index])
                axs[i].axis('off')
            else:
                plt.imshow(np_image)
                if titles:
                    plt.set_title(titles[index])
                plt.axis('off')

    plt.tight_layout()
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
