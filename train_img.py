import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import os
from torch.utils.data import Dataset, DataLoader

from dataset_loader_kernel import create_dataloaders, TestDataset, Train_img_Dataset
# from models.NKD_linear_layer import NKD_latent_layers, NKD_latent_layers_v2, save_checkpoint, load_checkpoint
from NKD_kenel_layer import NKD_kernel_layers_v1
from NKD_test import Test_denoisers


# args.seed, args.in_folder, args.out_folder, 
# args.color_mode:  'gray' or 'rgb', args.sigma_min, args.sigma_max, args.patch_size, 
# args.epochs, args.batch_size, args.lr, args.sch: 'multistep' or 'cosine', args.lr_step, args.lr_gamma, 
# args.val_freq, args.save_freq, 
# args.blind, args.patch_rad, args.window_rad, args.in_channel, args.latent_dim, args.latent_channel_dim
def train(args, verbose = False):
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainset = Train_img_Dataset('./data/img_train', args.color_mode, args.sigma_min, args.sigma_max, patch_size=args.patch_size, batch_size=args.batch_size)
    train_loader = DataLoader(
                trainset,
                batch_size=1,  # Process one full image at a time
                shuffle=False,
                pin_memory=True
            )

    # Create model
    in_nc = out_nc = 1 if args.color_mode == 'gray' else 3
    model = args.architecture(
        patch_rad=args.patch_rad,
        window_rad=args.window_rad,
        in_channel=in_nc,
        latent_dim=args.latent_dim,
        latent_channel_dim=args.latent_channel_dim,
        device=device,
        blind = args.blind
    ).to(device)

    # Setup optimizer, scheduler and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    # For tracking best validation results
    best_psnr = 0
    best_ssim = 0

    # Training loop
    total_steps = 0
    model.train()

    losses = []
    val_metric = []

    try:
        model.train()
        for epoch in range(args.epochs):
            batch_count = 0
            current_lr = optimizer.param_groups[0]['lr']

            loss = model.train_step(train_loader, criterion, optimizer, lamda=args.l1_reg_lamda)
            total_steps += 1
            batch_count += 1
            losses.append(loss)

            # Print progress
            if total_steps % args.progress_freq == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_count}/{len(train_loader)} | "
                        f"Loss: {loss:.6f} | lr = {current_lr}", end='\r')

            avg_loss = loss / batch_count

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise e
    finally:
        print("\nTraining complete!")
        print(f"Best PSNR: {best_psnr:.2f}dB")
        print(f"Best SSIM: {best_ssim:.4f}")

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, losses, val_metric

if __name__ == "__main__":
    # args.seed, args.in_folder, args.out_folder, 
    # args.color_mode:  'gray' or 'rgb', args.sigma_min, args.sigma_max, args.patch_size, 
    # args.epochs, args.batch_size, args.lr, args.sch: 'multistep', 'warmup_cosine' or 'cosine', args.lr_step, args.lr_gamma, 
    # args.val_freq, args.save_freq, args.progress_freq, args.save_img_freq,
    # args.blind, args.patch_rad, args.window_rad, args.in_channel, args.latent_dim, args.latent_channel_dim
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    class config:
        def __init__(self):
            pass

    args = config()

    # TRAINING PARAMETERS
    args.in_folder = './data'
    args.out_folder = './output'
    args.color_mode = 'grey'
    args.sigma_min = 30
    args.sigma_max = 30
    args.patch_size = 128
    args.epochs = 6000
    args.batch_size = 1
    args.lr = 3e-4
    args.sch = 'cosine'
    args.l1_reg_lamda = 0.00
    # #for warmup_cosine
    # args.num_warmup_steps = 200
    # args.num_cycles = 0.5
    # #for multistep
    # args.lr_step = None
    # args.lr_gamma = None

    # PROGRESS PARAMETERS
    args.val_freq = 100
    args.save_freq = 200
    args.save_img = True
    args.save_img_freq = 200
    args.progress_freq = 1

    # MODEL PARAMETERS
    args.architecture = NKD_kernel_layers_v1
    args.patch_rad = 3
    args.window_rad = 4
    args.in_channel = 1
    args.latent_dim = 128
    args.latent_channel_dim = 64
    args.blind = True

    # RESUME TRAINING ?
    args.resume_training = False
    args.checkpoint_path = ''

    verbose = False

    _ = train(args, verbose)

