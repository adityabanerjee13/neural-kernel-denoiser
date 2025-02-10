import torch
from datetime import datetime
from einops import repeat
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import os
from torch.utils.data import Dataset, DataLoader
from NKD_models import *

from dataset_loaders import Train_img_Dataset, TrainingDataset
# from models.NKD_linear_layer import NKD_latent_layers, NKD_latent_layers_v2, save_checkpoint, load_checkpoint
# from NKD_kenel_layer import NKD_kernel_layers_v1
# from NKD_test import Test_denoisers




def train(args, verbose = False):
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = TrainingDataset('./data/train', args.color_mode, args.sigma_min, args.sigma_max, 
                                 patch_size=args.patch_size, imgs_per_epoch=args.imgs_per_epoch,
                                 req_patches=args.req_patches, no_patches=args.no_patches)

    train_loader = DataLoader(
                trainset,
                batch_size=1,  # Process one full image at a time
                shuffle=False,
                pin_memory=True
            )
    
    # padding_dim = args.window_rad+args.patch_rad
    # testset = Train_img_Dataset('./data/img_train', args.color_mode, args.sigma_max, args.sigma_max, 
    #                              patch_size=2*(padding_dim)+args.test_imgsize, size=None,
    #                              train_subdirs=['img_baboon'],
    #                              imgs_per_epoch=1, req_patches=False)

    # test_loader = DataLoader(
    #     testset,
    #     batch_size=1,  # Process one full image at a time
    #     shuffle=False,
    #     pin_memory=True
    # )
    
    # Create model
    in_nc = out_nc = 1 if args.color_mode == 'gray' else 3
    model = architectures[args.architecture](args, device).to(device)

    # Get the current time
    time_stamp = datetime.now().strftime("%m%d%H%M%S")
    _state = f'_{str(model)}{args.sigma_min}_{args.sigma_max}_{time_stamp}'
    args.out_folder = args.out_folder+_state
    # Check if the output folder exists
    if not os.path.exists(args.out_folder):
        # Create the folder
        os.makedirs(args.out_folder)
        os.makedirs(os.path.join(args.out_folder, 'ckpt'))
        os.makedirs(os.path.join(args.out_folder, 'imgs'))
        if verbose:
            print(f"Folder created at: {args.out_folder}")

    # Setup optimizer, scheduler and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.sch == 'multistep':
        if isinstance(args.lr_step, str):
            milestones = [int(x) for x in args.lr_step.split(',')]
        else:
            milestones = [args.lr_step]
        # Multi-step learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma
        )
    elif args.sch == 'cosine':
        # Cosine annealing learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.sch == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.epochs,
            num_cycles=args.num_cycles
        )
    elif args.sch == 'const':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        raise ValueError(f"scehduler(={args.sch}) is not defined")
    criterion = nn.MSELoss()
    # For tracking best validation results
    best_psnr = 0
    best_ssim = 0

    # Training loop
    total_steps = 0
    model.train()

    losses = []
    val_metric = []

    # try:
    if args.resume_training:
        model, optimizer, scheduler, args = NKD_denoisers.load_checkpoint(args.checkpoint_path, device, optimizer, scheduler)

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
            print(f"Epoch {epoch+1}/{args.epochs} | "
                    f"Loss: {loss:.6f} | lr = {current_lr}", end='\r')

        # Regular checkpoint saving
        if total_steps % args.save_freq == 0:
            if verbose:
                print('saving checkpoint')
            model.save_checkpoint(args.out_folder, {
                    'step': total_steps,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'loss': loss,
                    'args': args,
                }, total_steps)

        # Save epoch loss to log
        with open(os.path.join(args.out_folder, 'training_log.txt'), 'a') as f:
            f.write(f'{epoch+1},{loss:.6f},{current_lr}\n')

        if args.sch!='const':
            scheduler.step()

        # # Save Image
        # if args.save_img and total_steps % args.save_img_freq == 0:
        #     TestDenoisers(test_loader, {'NKD : code' : model}, device, 
        #                   out_folder=os.path.join(args.out_folder, 'imgs'), 
        #                   p=padding_dim, state=f'{total_steps}', verbose=False)

    # except KeyboardInterrupt:
    #     print("\nTraining interrupted. Saving checkpoint...")

    # except Exception as e:
    #     print(f"\nERROR: {str(e)}")
    #     raise e

    # finally:
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

    # MODEL PARAMETERS
    # architecture = ['NKD_concat_kernel_learn']
    architecture = ['NKD_concat_kernel_learn']
    patch_rads = [5]
    args.window_rad = 7
    args.in_channel = 1
    args.latent_dim = 1024
    args.latent_channel_dim = 10
    args.residual_depth = 2
    proj_depths = [2,3,6]
    args.blind = True

    # TRAINING PARAMETERS
    args.in_folder = './data'
    args.out_folder = './output'
    args.color_mode = 'gray'
    args.sigma_min = 30
    args.sigma_max = 30
    args.req_patches = True
    args.patch_size = 128
    args.epochs = 50000
    args.batch_size = 1
    args.lr = 1e-4
    args.sch = 'const'
    args.l1_reg_lamda = 0.00
    # #for warmup_cosine
    # args.num_warmup_steps = 200
    # args.num_cycles = 0.5
    # #for multistep
    # args.lr_step = None
    # args.lr_gamma = None

    # PROGRESS DATA PARAMETERS
    args.val_freq = 100
    args.save_freq = 200
    args.save_img = True
    args.test_imgsize = 32
    args.save_img_freq = 200
    args.progress_freq = 1

    args.imgs_per_epoch = 1
    args.factor = None
    args.no_patches = 512

    # RESUME TRAINING ?
    args.resume_training = False
    args.checkpoint_path = ''
    verbose = False

    for proj_depth in proj_depths:
        for patch_rad in patch_rads:
            for arch in architecture:
                print(arch, patch_rad)
                args.out_folder = './output'
                args.proj_depth = proj_depth
                args.architecture = arch
                args.patch_rad = patch_rad
                if args.req_patches:
                    args.patch_size = 2*(args.window_rad+args.patch_rad)+1
                else:
                    pass
                _ = train(args, verbose)
