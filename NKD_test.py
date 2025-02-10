import torch
import torch.nn as nn
from deepinv.models import BM3D
import os

# from models.NKD_linear_layer import calculate_psnr, calculate_ssim, NKD_latent_layers, load_checkpoint
from NKD_kenel_layer import calculate_psnr, calculate_ssim, load_checkpoint
# from denoisers.NLM_torch import NLM_tensor
from utils import *
from dataset_loaders import TestDataset, DataLoader
class config:
    def __init__(self):
        pass




# def test(model : nn.Module, test_loader : DataLoader, device : str, state='', out_folder='./output_img'):
#     psnr_values = []
#     ssim_values = []

#     # Loop through the validation loader
#     for i, data in enumerate(test_loader):
#         print(f'Image:- {i}')

#         # fig, axes = plt.subplots(len(name), 3, figsize=(15, 5 * len(name)))

#         clean_image, noisy_image, sigma = data  # dataset returns a tuple of (noisy_image, clean_image)
        
#         # Move images to the same device as the model
#         noisy_image = noisy_image.to(device=device, dtype=torch.float32)
#         clean_image = clean_image.to(device=device, dtype=torch.float32)
#         sigma = sigma.to(device=device, dtype=torch.float32)

#         with torch.no_grad():
#             denoised_image = model(noisy_image, sigma)
            
#         # Calculate PSNR of denoised image
#         psnr = calculate_psnr(clean_image, denoised_image)
#         ssim = calculate_ssim(clean_image, denoised_image)
        


def Test_denoisers(test_loader : DataLoader, denoisers : dict, device : str, state='', out_folder='./output_img'):
    
    # assert len(denoisers) == len(name), "Number of denoisers and names should be same"
    print('==== Test ====')
    # Create a figure to save the images
    grid = (len(denoisers), 3)

    # Loop through the validation loader
    for i, data in enumerate(test_loader):
        print(f'Image:- {i}')

        # fig, axes = plt.subplots(len(name), 3, figsize=(15, 5 * len(name)))

        clean_image, noisy_image, sigma = data  # dataset returns a tuple of (noisy_image, clean_image)
        
        # Move images to the same device as the model
        noisy_image = noisy_image.to(device=device, dtype=torch.float32)
        clean_image = clean_image.to(device=device, dtype=torch.float32)
        sigma = sigma.to(device=device, dtype=torch.float32)

        psnr_noisy = calculate_psnr(clean_image, noisy_image)
        ssim_noisy = calculate_ssim(clean_image, noisy_image)
        print(f'Noisy: psnr = {psnr_noisy:.2f} | ssim = {ssim_noisy:.4f}')
        titles = []
        images = []

        for j, name in enumerate(denoisers.keys()):
            # Pass the noisy image through the model to get the denoised image
            if name=='BM3D':
                with torch.no_grad():
                    denoised_image = denoisers[name](noisy_image, sigma.squeeze().item())
                noisy_image= noisy_image.to(device)
                denoised_image = denoised_image.to(device)

            else:
                with torch.no_grad():
                    denoised_image = denoisers[name](noisy_image, sigma)
            
            # Calculate PSNR of denoised image
            psnr = calculate_psnr(clean_image, denoised_image)
            ssim = calculate_ssim(clean_image, denoised_image)
            # # calculate PSNR of noisy image
            print(f'{name}: psnr = {psnr:.2f} | ssim = {ssim:.4f}')
            
            images.append(noisy_image)
            titles.append(f'Noisy Image({psnr_noisy:.2f}dB)')

            images.append(denoised_image)
            titles.append(f'{name}: Denoised Image({psnr:.2f}dB)')

            images.append(clean_image)
            titles.append(f'Clean Image')

        show_tensor_image(images, grid, titles, save=True, save_path=os.path.join(out_folder, f'saved_test_{i}_{state}.png'))

def create_combined_checker_matrix(size=64, checker_sizes = [1, 3, 5, 7]):
    """Create a combined matrix with increasing checker pattern sizes"""
    combined_matrix = torch.zeros((size, size), dtype=torch.float32)
    
    # Divide the matrix into 4 quadrants
    quarter_size = size // 2
    
    for quad_row in range(2):
        for quad_col in range(2):
            checker_size = checker_sizes[quad_row * 2 + quad_col]
            
            # Define the region for this quadrant
            row_start = quad_row * quarter_size
            row_end = row_start + quarter_size
            col_start = quad_col * quarter_size
            col_end = col_start + quarter_size
            
            # Create checker pattern for this quadrant
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    if ((i // checker_size + j // checker_size) % 2 == 0):
                        combined_matrix[i, j] = 1
    
    return combined_matrix.unsqueeze(0)


def CheckerTestDenoisers(sigma, size, denoisers, device, out_folder, p=0, state='', verbose=False):
    if verbose:
        print('==== Test ====')
    # Create a figure to save the images
    grid = (len(denoisers), 3)

    clean_image = create_combined_checker_matrix(size=size, checker_sizes=[2**i for i in range(4)])
    noisy_image = clean_image + float(sigma)/255*torch.randn_like(clean_image)

    # Move images to the same device as the model
    clean_image = unpad(clean_image.to(device=device, dtype=torch.float32), p, p)
    noisy_image = noisy_image.to(device=device, dtype=torch.float32)
    sigma = sigma * torch.ones(1, 1, 1, device=device, dtype=torch.float32)
    unpadded_noisy_image = unpad(noisy_image, p,p)
    print(f'image size = {unpadded_noisy_image.shape}')

    psnr_noisy = calculate_psnr(clean_image, unpadded_noisy_image)
    ssim_noisy = calculate_ssim(clean_image, unpadded_noisy_image)
    if verbose:
        print(f'Noisy: psnr = {psnr_noisy:.2f} | ssim = {ssim_noisy:.4f}')
    titles = []
    images = []

    for j, name in enumerate(denoisers.keys()):
        # Pass the noisy image through the model to get the denoised image
        if name == 'BM3D':
            with torch.no_grad():
                denoised_image = denoisers[name](unpadded_noisy_image, sigma.squeeze().item())
            unpadded_noisy_image = unpadded_noisy_image.to(device)
            denoised_image = denoised_image.to(device)

        elif 'nkd' in name.lower():
            with torch.no_grad():
                denoised_image = denoisers[name].forward_img(noisy_image, sigma)
                if verbose:
                    print(denoised_image.shape, clean_image.shape)
        else:
            with torch.no_grad():
                denoised_image = denoisers[name](unpadded_noisy_image, sigma)
        
        # Calculate PSNR of denoised image
        if denoised_image.shape != clean_image.shape:
            h, w = denoised_image.shape[-2:]
            hc, wc = clean_image.shape[-2:]
            denoised_image = unpad(denoised_image, abs(h-hc)//2, abs(w-wc)//2)
        print(clean_image.shape, denoised_image.shape)
        psnr = calculate_psnr(clean_image, denoised_image)
        ssim = calculate_ssim(clean_image, denoised_image)
        if verbose:
            print(f'{name}: psnr = {psnr:.2f} | ssim = {ssim:.4f}')
        
        images.append(unpadded_noisy_image)
        titles.append(f'Noisy Image({psnr_noisy:.2f}dB)')

        images.append(denoised_image)
        titles.append(f'{name}: Denoised Image({psnr:.2f}dB)')

        images.append(clean_image)
        titles.append(f'Clean Image')

    show_tensor_image(images, grid, titles, save=True, save_path=os.path.join(out_folder, f'saved_test_checker_{state}.png'))

def TestDenoisers(data_loader, denoisers, device, out_folder, p=0, state='', verbose=False):
    if verbose:
        print('==== Test ====')
    # Create a figure to save the images
    grid = (len(denoisers), 3)

    # Loop through the validation loader
    for i, data in enumerate(data_loader):
        if verbose:
            print(f'Image:- {i}')

        clean_image, noisy_image, sigma = data  # dataset returns a tuple of (noisy_image, clean_image)
        
        # Move images to the same device as the model
        noisy_image = noisy_image.to(device=device, dtype=torch.float32).squeeze(0)
        unpadded_noisy_image = noisy_image[:, p:-p, p:-p]
        clean_image = clean_image.to(device=device, dtype=torch.float32).squeeze(0)[:, p:-p, p:-p]
        sigma = sigma.to(device=device, dtype=torch.float32)
        print(f'image size = {unpadded_noisy_image.shape}')

        psnr_noisy = calculate_psnr(clean_image, unpadded_noisy_image)
        ssim_noisy = calculate_ssim(clean_image, unpadded_noisy_image)
        if verbose:
            print(f'Noisy: psnr = {psnr_noisy:.2f} | ssim = {ssim_noisy:.4f}')
        titles = []
        images = []

        for j, name in enumerate(denoisers.keys()):
            # Pass the noisy image through the model to get the denoised image
            if name == 'BM3D':
                with torch.no_grad():
                    denoised_image = denoisers[name](unpadded_noisy_image, sigma.squeeze().item())
                unpadded_noisy_image = unpadded_noisy_image.to(device)
                denoised_image = denoised_image.to(device)

            elif 'nkd' in name.lower():
                with torch.no_grad():
                    denoised_image = denoisers[name].forward_img(noisy_image, sigma)
                    if verbose:
                        print(denoised_image.shape, clean_image.shape)
            else:
                with torch.no_grad():
                    denoised_image = denoisers[name](unpadded_noisy_image, sigma)
            
            if denoised_image.shape != clean_image.shape:
                h, w = denoised_image.shape[-2:]
                hc, wc = clean_image.shape[-2:]
                denoised_image = unpad(denoised_image, abs(h-hc)//2, abs(w-wc)//2)
            # Calculate PSNR of denoised image
            psnr = calculate_psnr(clean_image, denoised_image)
            ssim = calculate_ssim(clean_image, denoised_image)
            if verbose:
                print(f'{name}: psnr = {psnr:.2f} | ssim = {ssim:.4f}')
            
            images.append(unpadded_noisy_image)
            titles.append(f'Noisy Image({psnr_noisy:.2f}dB)')

            images.append(denoised_image)
            titles.append(f'{name}: Denoised Image({psnr:.2f}dB)')

            images.append(clean_image)
            titles.append(f'Clean Image')

        show_tensor_image(images, grid, titles, save=True, save_path=os.path.join(out_folder, f'saved_test_{i}_{state}.png'))


def main(color_mode='rgb'):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    testset = TestDataset('./data/show', 'rgb', 15, show=True)
    test_loader = DataLoader(
                testset,
                batch_size=1,  # Process one full image at a time
                shuffle=False,
                pin_memory=True
            )    
    # Create model
    in_nc = out_nc = 1 if color_mode == 'gray' else 3
    denoiser_nkd, _, _, _ = load_checkpoint('./output_510351210115_15/model_510351210115_15_step_9500.pth', device)
    denoiser_nkd.window_rad = 7

    # denoiser_nlm = lambda x, y : NLM_tensor(x, x, 1, 5, 5/255, False)
    denoiser_bm3d = BM3D()
    from PnP.regularizer import TV_regularizer
    denoiser_TV = TV_regularizer(lamda=0.025, iters=50, l=0, u=1, proj_p='isotropic')

    # denoisers = [denoiser_nkd.dsg_output, denoiser_nkd]
    # names = ['DSG-NKD', 'NKD']
    # denoisers=[denoiser_nkd, denoiser_nlm, denoiser_bm3d, denoiser_TV]
    # names=['NKD','NLM', 'BM3D', 'TV']
    denoisers=[denoiser_TV]
    names=['TV']

    # Test denoisers
    Test_denoisers(test_loader, denoisers, names, device)

# if __name__ == '__main__':
    
#     main()