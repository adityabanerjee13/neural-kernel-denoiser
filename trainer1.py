#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

import torch
import os


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model state from checkpoint: {checkpoint_path}")

    # Optionally load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state.")

    # Optionally load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state.")

    # Return other metadata like epoch, step, loss, etc.
    return {
        'step': checkpoint.get('step', None),
        'epoch': checkpoint.get('epoch', None),
        'loss': checkpoint.get('loss', None),
        'psnr': checkpoint.get('psnr', None),
        'ssim': checkpoint.get('ssim', None),
        'best_psnr': checkpoint.get('best_psnr', None),
        'best_ssim': checkpoint.get('best_ssim', None),
    }

def save_checkpoint(out_folder, state, step, is_best=False):
    """Helper function to save checkpoints"""
    checkpoint_path = os.path.join(out_folder, f'model_step{step}.pth')
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(out_folder, 'model_best.pth')
        torch.save(state, best_path)

def validate(model, val_loader, device, blind=False):
    """Run validation and compute metrics"""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    with torch.no_grad():
        for clean, noisy, sigma in val_loader:
            clean = clean.to(device, dtype=torch.float32)
            noisy = noisy.to(device, dtype=torch.float32)
            sigma = sigma.to(device, dtype=torch.float32)

            if blind:
                denoised = model(noisy)
            else:
                denoised = model(noisy, sigma)

            psnr = calculate_psnr(clean, denoised)
            ssim = calculate_ssim(clean, denoised)

            total_psnr += psnr
            total_ssim += ssim
            num_images += 1

    model.train()
    return total_psnr / num_images, total_ssim / num_images

def create_model(args, in_nc, out_nc):
    """Helper function to create model based on args"""
    if args.model_type == 'fdncnn':
        return FDnCNN(
            in_nc=in_nc,
            out_nc=out_nc,
            nb=args.nb,
            blind=args.blind,
            activation=args.activation,
            rescaling=args.rescaling,
            residual_type=args.residual_type,
            contractive_pos=args.contractive_position,
            contractive_residual=args.contractive_residual,
            val_niter=args.val_niter,
            mu=args.mu,
            cf=args.cf
        )
    else:  # drunet
        return DRUNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nb=args.nb,
            blind=args.blind,
            upscale_method=args.upscale_method,
            downscale_method=args.downscale_method,
            rescaling=args.rescaling,
            residual_type=args.residual_type,
            contractive_residual=args.contractive_residual,
            contractive_position=args.contractive_position,
            val_niter=args.val_niter,
            mu=args.mu,
            cf=args.cf
        )

def train(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu_id} is not available. Available GPUs: {
                             torch.cuda.device_count()}")
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create training dataset and dataloader
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    in_nc = out_nc = 1 if args.color_mode == 'gray' else 3
    model = create_model(args, in_nc, out_nc).to(device)

    # Setup optimizer, scheduler and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if isinstance(args.lr_step, str):
        milestones = [int(x) for x in args.lr_step.split(',')]
    else:
        milestones = [args.lr_step]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_gamma
    ) if args.sch == 'multistep' else optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    criterion = nn.MSELoss()
    # For tracking best validation results
    best_psnr = 0
    best_ssim = 0

    # Training loop
    total_steps = 0
    model.train()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    try:
        for epoch in range(args.epochs):
            epoch_loss = 0
            batch_count = 0

            for clean, noisy, sigma in train_loader:
                try:
                    clean = clean.to(device, dtype=torch.float32)
                    noisy = noisy.to(device, dtype=torch.float32)
                    sigma = sigma.to(device, dtype=torch.float32)

                    # More efficient than zero_grad()
                    optimizer.zero_grad(set_to_none=True)

                    if args.blind:
                        output = model(noisy)
                    else:
                        output = model(noisy, sigma)

                    loss = criterion(output, clean)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_steps += 1
                    batch_count += 1

                    # Validation
                    if val_loader is not None and total_steps % args.val_freq == 0:
                        model.eval()
                        with torch.autocast(device_type="cuda", dtype=torch.float32):
                            avg_psnr, avg_ssim = validate(
                                model, val_loader, device, args.blind)
                        model.train()

                        print(f"\nStep {total_steps} Validation:")
                        print(f"PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

                        # Track best results
                        is_best = False
                        if avg_psnr > best_psnr:
                            best_psnr = avg_psnr
                            is_best = True
                            print(f"New best PSNR: {best_psnr:.2f}dB")
                        if avg_ssim > best_ssim:
                            best_ssim = avg_ssim
                            is_best = True
                            print(f"New best SSIM: {best_ssim:.4f}")

                        # Save checkpoint
                        save_checkpoint(args.out_folder, {
                            'step': total_steps,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss.item(),
                            'psnr': avg_psnr,
                            'ssim': avg_ssim,
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'args': args,
                        }, total_steps, is_best)

                        # Log metrics
                        with open(os.path.join(args.out_folder, 'validation_log.txt'), 'a') as f:
                            f.write(f'{total_steps},{
                                    avg_psnr:.4f},{avg_ssim:.4f}\n')

                    # Regular checkpoint saving
                    elif total_steps % args.save_freq == 0:
                        save_checkpoint(args.out_folder, {
                            'step': total_steps,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss.item(),
                            'args': args,
                        }, total_steps)

                    # Print progress
                    if batch_count % 1000 == 0:
                        print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_count}/{len(train_loader)} | "
                              f"Loss: {loss.item():.6f}", end='\r')

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nWARNING: out of memory on batch {batch_count}. Skipping batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            scheduler.step()

            # Print epoch statistics
            avg_loss = epoch_loss / batch_count
            print(f"\nEpoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.6f}")
            
            # Save epoch loss to log
            with open(os.path.join(args.out_folder, 'training_log.txt'), 'a') as f:
                f.write(f'{epoch+1},{avg_loss:.6f}\n')

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(args.out_folder, {
            'step': total_steps,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / batch_count if batch_count > 0 else float('inf'),
            'device': str(device),
            'args': args,
        }, 'interrupted')
        print(f"Checkpoint saved to {args.out_folder}/model_interrupted.pth")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        save_checkpoint(args.out_folder, {
            'step': total_steps,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / batch_count if batch_count > 0 else float('inf'),
            'error': str(e),
            'args': args,
        }, 'error')
        print(f"Error checkpoint saved to {args.out_folder}/model_error.pth")
        raise e

    finally:
        end_time.record()
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(
            end_time) / 1000  # Convert to seconds

        print("\nTraining complete!")
        print(f"Total training time: {training_time/3600:.2f} hours")
        print(f"Best PSNR: {best_psnr:.2f}dB")
        print(f"Best SSIM: {best_ssim:.4f}")

        # Save final metrics
        with open(os.path.join(args.out_folder, 'final_metrics.txt'), 'w') as f:
            f.write(f"Training Time (hours): {training_time/3600:.2f}\n")
            f.write(f"Best PSNR (dB): {best_psnr:.2f}\n")
            f.write(f"Best SSIM: {best_ssim:.4f}\n")
            f.write(f"Final Loss: {
                    epoch_loss/batch_count if batch_count > 0 else float('inf'):.6f}\n")

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

