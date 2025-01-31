import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import os
from torch.utils.data import Dataset, DataLoader

from dataset_loaders import create_dataloaders, TestDataset
from models.NKD_linear_layer import NKD_latent_layers, NKD_latent_layers_v2, save_checkpoint, load_checkpoint
from NKD_kenel_layer import NKD_kernel_layers_v1, save_checkpoint, load_checkpoint
from NKD_test import Test_denoisers

# args.seed, args.in_folder, args.out_folder, 
# args.color_mode:  'gray' or 'rgb', args.sigma_min, args.sigma_max, args.patch_size, 
# args.epochs, args.batch_size, args.lr, args.sch: 'multistep' or 'cosine', args.lr_step, args.lr_gamma, 
# args.val_freq, args.save_freq, 
# args.blind, args.patch_rad, args.window_rad, args.in_channel, args.latent_dim, args.latent_channel_dim

def train(args, verbose = False):
    # Setup device
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Create training dataset and dataloader
    train_loader, val_loader = create_dataloaders(args)

    testset = TestDataset('./data/show', 'rgb', args.sigma_max, show=True)

    test_loader = DataLoader(
                testset,
                batch_size=1,  # Process one full image at a time
                shuffle=False,
                pin_memory=True
            )

    # Create model
    in_nc = out_nc = 1 if args.color_mode == 'gray' else 3
    model = args.architecture(
        patch_rad=args.patch_rad,
        window_rad=args.window_rad,
        in_channel=args.in_channel,
        latent_dim=args.latent_dim,
        latent_channel_dim=args.latent_channel_dim,
        device=device, 
        blind = args.blind
    ).to(device)

    _state = f'_{args.patch_rad}{args.window_rad}{args.in_channel}{args.latent_dim}{args.latent_channel_dim}{int(args.blind)}{args.sigma_min}_{args.sigma_max}'
    args.out_folder = args.out_folder+_state
    # Check if the output folder exists
    if not os.path.exists(args.out_folder):
        # Create the folder
        os.makedirs(args.out_folder)
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
    else:
        raise ValueError(f"scehduler(={args.sch}) is not defined")

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
    losses = []
    val_metric = []

    if args.resume_training:
        model, optimizer, scheduler, _ = load_checkpoint(args.checkpoint_path, args.architecture, device, optimizer, scheduler)
    # try:0
    try:
        model.train()
        for epoch in range(args.epochs):
            batch_count = 0
            current_lr = optimizer.param_groups[0]['lr']

            loss = model.train_step(train_loader, criterion, optimizer, lamda=args.l1_reg_lamda)
            total_steps += 1
            batch_count += 1
            losses.append(loss)

            # Validation
            if val_loader is not None and total_steps % args.val_freq == 0:
                model.eval()
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    mertics = model.val_step(val_loader)
                    avg_psnr, avg_ssim = mertics['psnr'], mertics['ssim']
                model.train()
                val_metric.append((avg_psnr, avg_ssim))
                if verbose:
                    print(f"\nStep {total_steps} Validation:")
                    print(f"PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")

                # Track best results
                is_best = False
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    is_best = True
                    if verbose:
                        print(f"New best PSNR: {best_psnr:.2f}dB")
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    is_best = True
                    if verbose:
                        print(f"New best SSIM: {best_ssim:.4f}")

                # Save checkpoint
                if is_best:
                    model.save_checkpoint(args.out_folder, {
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'loss': loss,
                        'psnr': avg_psnr,
                        'ssim': avg_ssim,
                        'best_psnr': best_psnr,
                        'best_ssim': best_ssim,
                        'args': args,
                    }, total_steps, True)

                # Log metrics
                with open(os.path.join(args.out_folder, 'validation_log.txt'), 'a') as f:
                    f.write(f'{total_steps},{
                            avg_psnr:.4f},{avg_ssim:.4f}\n')

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

            # Print progress
            if total_steps % args.progress_freq == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_count}/{len(train_loader)} | "
                        f"Loss: {loss:.6f} | lr = {current_lr}", end='\r')

            scheduler.step()

            if args.save_img and total_steps % args.save_img_freq == 0:
                Test_denoisers(test_loader, {'NKD' : model}, device, state=_state+f'_{total_steps}', out_folder=args.out_folder)

            avg_loss = loss / batch_count

            # Save epoch loss to log
            with open(os.path.join(args.out_folder, 'training_log.txt'), 'a') as f:
                f.write(f'{epoch+1},{avg_loss:.6f}\n')

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save_checkpoint(args.out_folder, {
            'step': total_steps,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss / batch_count if batch_count > 0 else float('inf'),
            'device': str(device),
            'args': args,
        }, 'interrupted')
        print(f"Checkpoint saved to {args.out_folder}/model_interrupted.pth")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        model.save_checkpoint(args.out_folder, {
            'step': total_steps,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss / batch_count if batch_count > 0 else float('inf'),
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
                    loss/batch_count if batch_count > 0 else float('inf'):.6f}\n")

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
    args.color_mode = 'rgb'
    args.sigma_min = 30
    args.sigma_max = 30
    args.patch_size = 16
    args.epochs = 6000
    args.batch_size = 16
    args.lr = 1e-4
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
    # args.architecture = NKD_li
    args.patch_rad = 5
    args.window_rad = 10
    args.in_channel = 3
    args.latent_dim = 256
    args.latent_channel_dim = 64
    args.blind = True

    # RESUME TRAINING ?
    args.resume_training = False
    args.checkpoint_path = ''

    verbose = False

    _ = train(args, verbose)












# def train(args, verbose = False):
#     except KeyboardInterrupt:
#         print("\nTraining interrupted. Saving checkpoint...")
#         save_checkpoint(args.out_folder, {
#             'step': total_steps,
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'loss': loss / batch_count if batch_count > 0 else float('inf'),
#             'device': str(device),
#             'args': args,
#         }, 'interrupted')
#         print(f"Checkpoint saved to {args.out_folder}/model_interrupted.pth")
#     except Exception as e:
#         print(f"\nERROR: {str(e)}")
#         save_checkpoint(args.out_folder, {
#             'step': total_steps,
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'loss': loss / batch_count if batch_count > 0 else float('inf'),
#             'error': str(e),
#             'args': args,
#         }, 'error')
#         print(f"Error checkpoint saved to {args.out_folder}/model_error.pth")
#         raise e

#     finally:
#         end_time.record()
#         torch.cuda.synchronize()
#         training_time = start_time.elapsed_time(
#             end_time) / 1000  # Convert to seconds

#         print("\nTraining complete!")
#         print(f"Total training time: {training_time/3600:.2f} hours")
#         print(f"Best PSNR: {best_psnr:.2f}dB")
#         print(f"Best SSIM: {best_ssim:.4f}")

#         # Save final metrics
#         with open(os.path.join(args.out_folder, 'final_metrics.txt'), 'w') as f:
#             f.write(f"Training Time (hours): {training_time/3600:.2f}\n")
#             f.write(f"Best PSNR (dB): {best_psnr:.2f}\n")
#             f.write(f"Best SSIM: {best_ssim:.4f}\n")
#             f.write(f"Final Loss: {
#                     loss/batch_count if batch_count > 0 else float('inf'):.6f}\n")

#         # Clean up
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         return model, losses, val_metric

