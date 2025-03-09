import torch
import torch.nn as nn
import os
from einops import repeat
from kernel_prediction_network import KPN
from utils import *
import math



def l1_loss(model):
    """
    Compute L1 regularization loss for a model's parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        l1_loss: Sum of absolute values of all model parameters
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += abs(param).sum()
    return l1_loss

class NKD_denoisers(nn.Module):
    def _extract_patch_vector(self, inp):
        batch, channel, _, _ = inp.shape
        patch_size = 2*self.patch_rad + 1
        t = torch.zeros(channel*patch_size*patch_size, channel, patch_size, patch_size, device=inp.device, dtype=inp.dtype)
        count=0
        for c in range(channel):
            for i in range(patch_size):
                for j in range(patch_size):
                    t[count,c,i,j] = 1
                    count+=1

        return torch.functional.F.conv2d(inp, t)

    # Defined in child class
    def pre_feature(self,inp_img):
        pass

    # Defined in child class
    def compute_activation(self,inp_img, t):
        pass

    # Defined in child class
    def _pre_process_input(self):
        # to be defined in the child class
        pass

    # Defined in child class
    def _get_activation(self):
        # to be defined in the child class
        pass

    def forward(self, inp, sig=None): 
        # print(torch.max(inp))
        # print(inp.shape)
        if len(inp.shape)<4:
            inp = inp.unsqueeze(0)
        b, c, h, w = inp.shape

        guide = 2 * inp.clone() - 1
        b, _, h, w = guide.shape
        if self.blind:
            pass
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            if hasattr(self, 'sigma_min'):
                sig = (sig-self.sigma_min)/(self.sigma_max-self.sigma_min)
                sig = 2*sig-1
            guide = torch.cat(
                (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                dim = 1
            )

        # Extract guide weights
        # print("guide",guide.shape)
        guide = self._pre_process_input(guide)
        # print("guide",guide.shape)
        guide = self._get_activation(guide)
        # print("guide_weights",guide.shape)
        normalizing_factor = torch.sum(guide, dim=1)
        # print("normalizing_factor",normalizing_factor.shape)

        inp = img2channel_list(unpad(inp, self.patch_rad, self.patch_rad))
        # print("inp",inp.shape)
        out = torch.sum((inp*guide), dim=1)/normalizing_factor
        # print("out",out.shape)

        return out.reshape(b, self.in_channel)

    def forward_img(self, img, guide=None, sig=None, span=32):
        self.eval()
        # Extract patches from image window
        if guide==None:
            guide = img.clone()
        p = self.window_rad+self.patch_rad
        
        if len(img.shape) == 4:
            img = img.squeeze(0)
            if len(img.shape) == 4:
                raise NotImplementedError('forward_img is not implemented for batches')
            else:
                return self.forward_img(img, guide, sig=sig, span=span).unsqueeze(0)
        elif len(img.shape) == 3:
            # print(img.shape)
            output = unpad(torch.zeros_like(img), p, p)
            # print(output.shape)
            c, h, w = img.shape
            for j in range(p, w-p, span):
                for i in range(p, h-p, span):
                    _, h_o, w_o = output[:, i-p:i-p+span, j-p:j-p+span].shape
                    i_str, i_stp = i-p, i+span+p
                    j_str, j_stp = j-p, j+span+p
                    # print(i_str, i_stp, j_str, j_stp)
                    # print(i, j)
                    img_patches = extract_all_patches(img[:, i_str:i_stp, j_str:j_stp], (self.window_rad+self.patch_rad))
                    # print(img_patches.shape, img[:, i_str:i_stp, j_str:j_stp].shape)
                    with torch.no_grad():
                        out_channel_list = self(img_patches, sig=sig)
                    # print(out_channel_list.shape, i, j)#, output[:, i-p:i-p+span, j-p:j-p+span].shape, channel_list2img(out_channel_list, h_o, w_o).shape)
                    output[:, i-p:i-p+span, j-p:j-p+span] = channel_list2img(out_channel_list, h_o, w_o)
        return output

    # def forward_img(self, img, guide, sig=None, span=32):
    #     self.eval()
    #     # Extract patches from image window
    #     p = self.window_rad+self.patch_rad
        
    #     if len(img.shape) == 4:
    #         img = img.squeeze(0)
    #         if len(img.shape) == 4:
    #             raise NotImplementedError('forward_img is not implemented for batches')
    #         else:
    #             return self.forward_img(img, guide, span=span).unsqueeze(0)
    #     elif len(img.shape) == 3:
    #         # print(img.shape)
    #         output = unpad(torch.zeros_like(img), p, p)
    #         # print(output.shape)
    #         c, h, w = img.shape
    #         for j in range(p, w-p, span):
    #             for i in range(p, h-p, span):
    #                 _, h_o, w_o = output[:, i-p:i-p+span, j-p:j-p+span].shape
    #                 i_str, i_stp = i-p, i+span+p
    #                 j_str, j_stp = j-p, j+span+p
    #                 print(i_str, i_stp, j_str, j_stp)
    #                 print(i, j)
    #                 img_patches = extract_all_patches(img[:, i_str:i_stp, j_str:j_stp], (self.window_rad+self.patch_rad))
    #                 guide_patches = extract_all_patches(guide[:, i_str:i_stp, j_str:j_stp], (self.window_rad+self.patch_rad))
    #                 # print(img_patches.shape, img[:, i_str:i_stp, j_str:j_stp].shape)
    #                 with torch.no_grad():
    #                     guide_patches = 2 * guide_patches.clone() - 1
    #                     b, _, h, w = guide_patches.shape
    #                     if self.blind:
    #                         pass
    #                     else:
    #                         assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
    #                         guide_patches = torch.cat(
    #                             (guide_patches, torch.ones((b,1,h,w), device=self.device, dtype=guide_patches.dtype)*sig),
    #                             dim = 1
    #                         )
    #                     guide_patches = self._pre_process_input(guide_patches)
    #                     # print("guide",guide.shape)
    #                     guide_patches = self._get_activation(guide_patches)
    #                     # print("guide_weights",guide_weights)
    #                     normalizing_factor = torch.sum(guide_patches, dim=1)

    #                     img_patches = img2channel_list(unpad(img_patches, self.patch_rad, self.patch_rad))
    #                     out_channel_list = torch.sum((img_patches*guide_patches), dim=1)/normalizing_factor
    #                     # out_channel_list = self(img_patches, sig=None)[0]
    #                 # print(out_channel_list.shape, i, j, output[:, i-p:i-p+span, j-p:j-p+span].shape, channel_list2img(out_channel_list, h_o, w_o).shape)
    #                 output[:, i-p:i-p+span, j-p:j-p+span] = channel_list2img(out_channel_list, h_o, w_o)
    #     return output

    def NLM(self, input_image, guide_features, sig=None):
        window_rad = self.window_rad
        patch_rad = self.patch_rad
        
        input_image = input_image.to(torch.float32)
        guide = self.pre_feature(
            pad(guide_features, patch_rad), 
            sig=sig
        )
        _, _, height, width = input_image.shape

        U = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold denoised image
        Z = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold accumulated weights

        padded_v = pad(input_image, window_rad)
        padded_guide = pad(guide, window_rad)

        tem = 2*window_rad+1
        T = tem**2
        cum_time=0
        
        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                t = padded_guide[:, :, self.window_rad+dx:self.window_rad+dx+height, self.window_rad+dy:self.window_rad+dy+width]
                activation = self.compute_activation(guide, t)
                hat = Box(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)

                w = hat * activation
                v = padded_v[:, :, window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                U = U + w * v
                Z = Z + w
        U = U / Z
        return U.to(torch.float32)  # U = Denoised image

    def train_step(self, train_loader, criterion, optimizer, lamda = 1e-3):
        self.train()
        avg_loss = 0
        count = 0

        for clean, noisy, sigma in train_loader:
            clean = clean.to(self.device, dtype=torch.float32)
            noisy = noisy.to(self.device, dtype=torch.float32)
            sigma = sigma.to(self.device, dtype=torch.float32)
            if len(noisy.shape)>4:
                c, h, w = noisy.size()[-3:]
                noisy= noisy.reshape(-1,c,h,w)

                if len(clean.shape)>2:
                    clean = clean.reshape(-1,c)
                if len(sigma.shape)>4:
                    sigma = sigma.reshape(-1,1,1,1)

            # print(sigma.shape, noisy.shape)
            # More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            output = self(noisy, sigma)

            l2_loss_act = 0

            loss = criterion(output, clean)
            loss_t = loss + l2_loss_act
            loss_t.backward()
            optimizer.step()

            b = output.size(0)
            count+=b
            avg_loss += (loss.item() * b)

        avg_loss = avg_loss/count
        if avg_loss>100:
            raise ValueError('gone')
        return avg_loss

    def val_step(self, val_loader):
        self.eval()
        metric = {'psnr' : 0, 'ssim' : 0}

        with torch.no_grad():
            for clean, noisy, sigma in val_loader:
                clean = clean.to(self.device, dtype=torch.float32)
                noisy = noisy.to(self.device, dtype=torch.float32)
                sigma = sigma.to(self.device, dtype=torch.float32)
                b = clean.size(0)

                if len(noisy.shape)>4:
                    c, h, w = noisy.size()[-3:]
                    noisy= noisy.reshape(-1,c,h,w)
                
                if len(clean.shape)>2:
                    clean= clean.reshape(-1,c)
            
                if self.blind:
                    output = self(noisy)
                else:
                    output = self(noisy, sigma)

                metric['psnr'] += calculate_psnr(output, clean)
                metric['ssim'] += calculate_ssim(output, clean)

        metric['ssim'] = metric['ssim']/len(val_loader)
        metric['psnr'] = metric['psnr']/len(val_loader)
        return metric

    def save_checkpoint(self, out_folder, state, step, is_best=False, _state=''):
        """Helper function to save checkpoints"""
        if state == '':
            # _state = f'_{str(self)}{state['args'].sigma_min}_{state['args'].sigma_max}'
            _state = f'{step}'
        else:
            pass
        if is_best:
            checkpoint_path = os.path.join(os.path.join(out_folder,'ckpt'), f'model{_state}_best.pth')
        else:
            checkpoint_path = os.path.join(os.path.join(out_folder,'ckpt'), f'model{_state}_step_{step}.pth')
        torch.save(state, checkpoint_path) 
    
    @staticmethod
    def load_checkpoint(checkpoint_path, device, optimizer=None, scheduler=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

        # Load the checkpoint
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = architectures[state['args'].architecture](state['args'], device).to(device)

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
        return model, state['args'], optimizer, scheduler

class NKD_concat_kernel_learn(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel

        if args.blind:
            in_dim=2*args.in_channel*patch_size*patch_size
        else:
            in_dim=2*(args.in_channel+1)*patch_size*patch_size
            
        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)
        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*4,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])
        # self.in_proj = nn.Linear(in_features=in_dim,out_features=latent_dim)

        self.proj = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim//(2**i),out_features=args.latent_dim//(2**(i+1))),
                nn.LayerNorm(args.latent_dim//(2**(i+1))),
                nn.LeakyReLU(),
            )
            for i in range(args.proj_depth)
        ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel)

    def _concat_patch_central_pixel(self, inp):
        b, c, h, w = inp.shape
        central_pixel = img2channel_list((inp[:,:,h//2,w//2])[:,:,None,None].expand_as(inp))
        channel_list = img2channel_list(inp)
        return (torch.cat((channel_list,central_pixel), dim=-1), 
                torch.cat((central_pixel,channel_list), dim=-1))

    def _get_activation(self, x):
        # print('xa', x.shape)
        x = self.in_layer(x)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
            # print('xc', x)
        # x = self.in_proj(x)
        x = self.proj(x)
        # print('xd', x.shape)
        x = self.out(x)
        # print('xe', x)
        x1,x2 = torch.chunk(x, chunks=2, dim=1)
        # print('x1', x1)
        return torch.exp((x1+x2)/(-2*self.sigma))

    def _pre_process_input(self, inp):
        # print("inp",inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x",x.shape)
        x1, x2 = self._concat_patch_central_pixel(x)
        # print("x1",x1.shape)
        # print("x2",x2.shape)
        inp = torch.cat((x1,x2), dim=1)
        # print("inp",inp.shape)
        return inp

    def pre_feature(self,inp_img):
        inp_img = 2 * inp_img.clone() - 1
        x = self._extract_patch_vector(inp_img)
        return x

    def _get_activation_img(self, x):
        # print('xa', x.shape)
        x = linear_img_fwd(x, self.in_layer)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + blk_img_fwd(x, self.in_residual[i])
            x = norm_img_fwd(x, self.in_norm[i])
            # print('xc', x)
        x = blk_img_fwd(x, self.proj)
        # print('xd', x.shape)
        x = linear_img_fwd(x, self.out)
        # print('xe', x.shape)
        return x

    def compute_activation(self,inp_img, t):
        x1, x2 = (torch.cat((inp_img, t), dim=1), torch.cat((t, inp_img), dim=1))
        x = torch.cat((x1,x2), dim=-1)
        # print('NLM-guide',x)
        x = self._get_activation_img(x)
        # print('NLM-out',x.shape)
        x1,x2 = torch.chunk(x, chunks=2, dim=-1)
        # print('NLM-out',x1.shape)
        # print('NLM-out',x2.shape)
        return torch.exp((x1+x2)/(-2*self.sigma))

    def __str__(self):
        return '_concat_kl'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

class NKD_concat_kernel_learn_v2(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel

        if args.blind:
            in_dim=2*args.in_channel*patch_size*patch_size
        else:
            in_dim=2*(args.in_channel+1)*patch_size*patch_size
            
        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*4,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])

        # if args.pre_res:
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim//(2**i),out_features=args.latent_dim//(2**(i+1))),
                nn.LayerNorm(args.latent_dim//(2**(i+1))),
                nn.LeakyReLU(),
            )
            for i in range(args.proj_depth)
        ])
        self.proj_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=(args.latent_dim//(2**(i+1))),out_features=(args.latent_dim//(2**(i+1)))*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=(args.latent_dim//(2**(i+1)))*4,out_features=(args.latent_dim//(2**(i+1)))),
            )
            for i in range(args.proj_depth)
        ])
        self.proj_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim//(2**(i+1)))
            for i in range(args.proj_depth)
        ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel)

    def _concat_patch_central_pixel(self, inp):
        b, c, h, w = inp.shape
        central_pixel = img2channel_list((inp[:,:,h//2,w//2])[:,:,None,None].expand_as(inp))
        channel_list = img2channel_list(inp)
        return (torch.cat((channel_list,central_pixel), dim=-1), 
                torch.cat((central_pixel,channel_list), dim=-1))

    def _get_activation(self, x):
        # print('xa', x.shape)
        x = self.in_layer(x)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
            # print('xc', x)

        for i in range(self.proj_depth):
            x = self.proj[i](x)
            x = x + self.proj_residual[i](x)
            x = self.proj_norm[i](x)
            # print('xc', x)

        # print('xd', x.shape)
        x = self.out(x)
        # print('xe', x)
        x1,x2 = torch.chunk(x, chunks=2, dim=1)
        # print('x1', x1)
        return torch.exp((x1+x2)/(-2*self.sigma))

    def _pre_process_input(self, inp):
        # print("inp",inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x",x.shape)
        x1, x2 = self._concat_patch_central_pixel(x)
        # print("x1",x1.shape)
        # print("x2",x2.shape)
        inp = torch.cat((x1,x2), dim=1)
        # print("inp",inp.shape)
        return inp

    def pre_feature(self,inp_img):
        inp_img = 2 * inp_img.clone() - 1
        x = self._extract_patch_vector(inp_img)
        return x

    def _get_activation_img(self, x):
        # print('xa', x.shape)
        x = linear_img_fwd(x, self.in_layer)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + blk_img_fwd(x, self.in_residual[i])
            x = norm_img_fwd(x, self.in_norm[i])
            # print('xc', x)
        for i in range(self.proj_depth):
            x = blk_img_fwd(x, self.proj[i])
            x = x + blk_img_fwd(x, self.proj_residual[i])
            x = norm_img_fwd(x, self.proj_norm[i])
        # print('xd', x.shape)
        x = linear_img_fwd(x, self.out)
        # print('xe', x.shape)
        return x

    def compute_activation(self,inp_img, t):
        x1, x2 = (torch.cat((inp_img, t), dim=1), torch.cat((t, inp_img), dim=1))
        x = torch.cat((x1,x2), dim=-1)
        # print('NLM-guide',x)
        x = self._get_activation_img(x)
        # print('NLM-out',x.shape)
        x1,x2 = torch.chunk(x, chunks=2, dim=-1)
        # print('NLM-out',x1.shape)
        # print('NLM-out',x2.shape)
        return torch.exp((x1+x2)/(-2*self.sigma))

    def __str__(self):
        return '_concat_kl_v2'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

def l2_loss_activation(x):
    return torch.mean(torch.sum(x**2, dim=-1))

class NKD_concat_kernel_learn_v3(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel
        self.proj_res_flag = args.proj_res_flag

        if args.blind:
            in_dim=args.in_channel*patch_size*patch_size
        else:
            in_dim=(args.in_channel+1)*patch_size*patch_size
            
        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*4,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])

        # if args.pre_res:
        self.proj = nn.ModuleList([])
        for i in range(args.proj_depth):
            in_dim = args.latent_dim//(2**i) if i!=0 else 2*args.latent_dim//(2**i)
            self.proj.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim,out_features=args.latent_dim//(2**(i+1))),
                    nn.LayerNorm(args.latent_dim//(2**(i+1))),
                    nn.LeakyReLU(),
                )
            )
        if self.proj_res_flag:
            self.proj_residual = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1))),out_features=(args.latent_dim//(2**(i+1)))*4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1)))*4,out_features=(args.latent_dim//(2**(i+1)))),
                )
                for i in range(args.proj_depth)
            ])
            self.proj_norm = nn.ModuleList([
                nn.LayerNorm(args.latent_dim//(2**(i+1)))
                for i in range(args.proj_depth)
            ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel)

    def _concat_patch_central_pixel(self, inp):
        central_pixel = inp[:,inp.size(1)//2,:][:,None,:].expand_as(inp)
        return (torch.cat((inp,central_pixel), dim=-1), 
                torch.cat((central_pixel,inp), dim=-1))

    def _get_pre_activation(self, x):
        # print('xa', x.shape)
        x = self.in_layer(x)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
            # print('xc', x)
        return x

    def _get_activation(self, x):
        x = self._concat_patch_central_pixel(x)
        x = torch.cat(x, dim=1)

        for i in range(self.proj_depth):
            x = self.proj[i](x)
            if self.proj_res_flag:
                x = x + self.proj_residual[i](x)
                x = self.proj_norm[i](x)
            # print('xc', x)

        # print('xd', x.shape)
        x = self.out(x)
        # print('xe', x)
        x1,x2 = torch.chunk(x, chunks=2, dim=1)
        # print('x1', x1)
        x = (x1+x2)/(-2*self.sigma)
        # return torch.exp(x), l2_loss_activation(x)
        return torch.functional.F.relu(x)+1e-4
        # return torch.square((x1+x2)/(-2*self.sigma)) + 1e-8

    def _pre_process_input(self, inp):
        # print("inp", inp, inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x", x, x.shape)
        x = img2channel_list(x)
        # print("x", x, x.shape)
        inp = self._get_pre_activation(x)
        return inp

    def _get_activation_img(self, x, pre=False):
        if pre:
            x = linear_img_fwd(x, self.in_layer)
            # print('xb', x.shape)
            for i in range(self.residual_depth):
                x = x + blk_img_fwd(x, self.in_residual[i])
                x = norm_img_fwd(x, self.in_norm[i])
                # print('xc', x)
            return x
        else:
            for i in range(self.proj_depth):
                x = blk_img_fwd(x, self.proj[i])
                if self.proj_res_flag:
                    x = x + blk_img_fwd(x, self.proj_residual[i])
                    x = norm_img_fwd(x, self.proj_norm[i])
            # print('xd', x.shape)
            x = linear_img_fwd(x, self.out)
            # print('xe', x.shape)
            return x

    def pre_feature(self,inp_img, sig):
        inp_img = 2 * inp_img.clone() - 1
        if self.blind:
            pass
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            b, _, h, w = inp_img.shape
            inp_img = torch.cat(
                (inp_img, torch.ones((b,1,h,w), device=self.device, dtype=inp_img.dtype)*sig),
                dim = 1
            )
        x = self._extract_patch_vector(inp_img)
        x = self._get_activation_img(x, pre=True)
        return x

    def compute_activation(self,inp_img, t):
        x1, x2 = (torch.cat((inp_img, t), dim=1), torch.cat((t, inp_img), dim=1))
        x = torch.cat((x1,x2), dim=-1)
        # print('NLM-guide',x)
        x = self._get_activation_img(x)
        # print('NLM-out',x.shape)
        x1,x2 = torch.chunk(x, chunks=2, dim=-1)
        # print('NLM-out',x1.shape)
        # print('NLM-out',x2.shape)
        # return torch.square((x1+x2)/(-2*self.sigma)) + 1e-8
        return torch.exp((x1+x2)/(-2*self.sigma))

    def __str__(self):
        return '_concat_kl_v3'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

class NKD_concat_kernel_learn_v4(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel
        self.proj_res_flag = args.proj_res_flag
        # if hasattr(args, 'pos_emb'):
        if hasattr(args, 'pos_emb'):
            self.pos_emb = args.pos_emb

        else:
            self.pos_emb = False

        if args.blind:
            in_dim=args.in_channel*patch_size*patch_size
        else:
            in_dim=(args.in_channel+1)*patch_size*patch_size
            
        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*4,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])

        # if args.pre_res:
        self.proj = nn.ModuleList([])
        for i in range(args.proj_depth):
            in_dim = args.latent_dim//(2**i) if i!=0 else 2*args.latent_dim//(2**i)
            self.proj.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim,out_features=args.latent_dim//(2**(i+1))),
                    nn.LayerNorm(args.latent_dim//(2**(i+1))),
                    nn.LeakyReLU(),
                )
            )
        if self.proj_res_flag:
            self.proj_residual = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1))),out_features=(args.latent_dim//(2**(i+1)))*4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1)))*4,out_features=(args.latent_dim//(2**(i+1)))),
                )
                for i in range(args.proj_depth)
            ])
            self.proj_norm = nn.ModuleList([
                nn.LayerNorm(args.latent_dim//(2**(i+1)))
                for i in range(args.proj_depth)
            ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel, bias=False)
        

        in_dim = 2*args.latent_dim
        latent = args.latent_dim//2
        self.proj_alpha = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=latent),
            nn.LayerNorm(latent),
            nn.LeakyReLU(),
        )
        self.proj_res_alpha = nn.Sequential(
            nn.Linear(in_features=(latent),out_features=(latent)*4),
            nn.LeakyReLU(),
            nn.Linear(in_features=(latent)*4,out_features=(latent)),
        )
        self.proj_norm_alpha = nn.LayerNorm(latent)
        self.out_alpha = nn.Linear(in_features=latent,out_features=args.in_channel, bias=False)


        self.proj_beta = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=latent),
            nn.LayerNorm(latent),
            nn.LeakyReLU(),
        )
        self.proj_res_beta = nn.Sequential(
            nn.Linear(in_features=(latent),out_features=(latent)*4),
            nn.LeakyReLU(),
            nn.Linear(in_features=(latent)*4,out_features=(latent)),
        )
        self.proj_norm_beta = nn.LayerNorm(latent)
        self.out_beta = nn.Linear(in_features=latent,out_features=args.in_channel, bias=False)

    def _concat_patch_central_pixel(self, inp):
        central_pixel = inp[:,inp.size(1)//2,:][:,None,:].expand_as(inp)
        return (torch.cat((inp,central_pixel), dim=-1), 
                torch.cat((central_pixel,inp), dim=-1))

    def _get_pre_activation(self, x, shape):
        # print('xa', x.shape)
        x = self.in_layer(x)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
            # print('xc', x.shape)
        if self.pos_emb:
            x = x + img2channel_list(get_patch_position_embedding(x.size(-1), shape, device= x.device)).repeat(x.size(0),1,1)
        return x
    
    def proj_function(self, x, value='k'):
        if value=='k':
            for i in range(self.proj_depth):
                x = self.proj[i](x)
                if self.proj_res_flag:
                    x = x + self.proj_residual[i](x)
                    x = self.proj_norm[i](x)
        elif value=='alpha':
            x = self.proj_alpha(x)
            x = x + self.proj_res_alpha(x)
            x = self.proj_norm_alpha(x)
        elif value=='beta':
            x = self.proj_beta(x)
            x = x + self.proj_res_beta(x)
            x = self.proj_norm_beta(x)
        else:
            raise ValueError(f'{value} is not implimented.')
        return x

    def controlled_sigmoid(self, x, a, b):
        # a(x) = Relu/sq(alpha) / ( 1 + exp ( - sigmoid (beta) x)
        return a/(1+torch.exp(-b*x))

    def _get_activation(self, x):
        x = self._concat_patch_central_pixel(x)
        x1 = torch.cat(x, dim=1)

        # print(x1.shape)
        x = self.proj_function(x1, value='k')
        # print(x.shape)
        k = self.out(x)
        k1,k2 = torch.chunk(k, chunks=2, dim=1)
        k = (k1+k2)/(2*self.sigma)

        # print(x1.shape)
        x = self.proj_function(x1, value='alpha')
        # print(x.shape)
        a = self.out_alpha(x)
        a1,a2 = torch.chunk(a, chunks=2, dim=1)
        a = torch.functional.F.relu((a1+a2)/2)+1e-4

        # print(x1.shape)
        x = self.proj_function(x1, value='beta')
        # print(x.shape)
        b = self.out_alpha(x)
        # print(b.shape)
        b1,b2 = torch.chunk(b, chunks=2, dim=1)
        # print(b1.shape)
        b = torch.functional.F.sigmoid((b1+b2)/2)

        return self.controlled_sigmoid(k, a, b)
        # return torch.exp((x1+x2)/(-2*self.sigma))
        # return torch.square((x1+x2)/(-2*self.sigma)) + 1e-8

    def _pre_process_input(self, inp):
        # print("inp", inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x", x.shape)
        h, w = x.shape[-2:]
        x = img2channel_list(x)
        # print("x", x.shape)
        inp = self._get_pre_activation(x, (h,w))
        return inp

    def _get_activation_img(self, x, pre=False):
        if pre:
            x = linear_img_fwd(x, self.in_layer)
            # print('xb', x.shape)
            for i in range(self.residual_depth):
                x = x + blk_img_fwd(x, self.in_residual[i])
                x = norm_img_fwd(x, self.in_norm[i])
                # print('xc', x)
            return x
        else:
            for i in range(self.proj_depth):
                x = blk_img_fwd(x, self.proj[i])
                if self.proj_res_flag:
                    x = x + blk_img_fwd(x, self.proj_residual[i])
                    x = norm_img_fwd(x, self.proj_norm[i])
            # print('xd', x.shape)
            x = linear_img_fwd(x, self.out)
            # print('xe', x.shape)
            return x

    def pre_feature(self,inp_img):
        inp_img = 2 * inp_img.clone() - 1
        x = self._extract_patch_vector(inp_img)
        x = self._get_activation_img(x, pre=True)
        return x

    def compute_activation(self,inp_img, t):
        x1, x2 = (torch.cat((inp_img, t), dim=1), torch.cat((t, inp_img), dim=1))
        x = torch.cat((x1,x2), dim=-1)
        # print('NLM-guide',x)
        x = self._get_activation_img(x)
        # print('NLM-out',x.shape)
        x1,x2 = torch.chunk(x, chunks=2, dim=-1)
        # print('NLM-out',x1.shape)
        # print('NLM-out',x2.shape)
        # return torch.square((x1+x2)/(-2*self.sigma)) + 1e-8
        return torch.exp((x1+x2)/(-2*self.sigma))

    def __str__(self):
        return 'concat_kl_v4'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

class NKD_concat_kernel_learn_v5(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad*2 + 1
        self.output_activation = args.output_activation

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel
        self.proj_res_flag = args.proj_res_flag
        self.pos_emb = args.pos_emb
        self.sigma_max = args.sigma_max/255
        self.sigma_min = args.sigma_min/255

        if args.blind:
            in_dim=args.in_channel*patch_size*patch_size
        else:
            in_dim=(args.in_channel+1)*patch_size*patch_size

        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)

        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*4),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*4,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])

        # if args.pre_res:
        self.proj = nn.ModuleList([])
        for i in range(args.proj_depth):
            in_dim = args.latent_dim//(2**i) if i!=0 else 2*args.latent_dim//(2**i)
            self.proj.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim,out_features=args.latent_dim//(2**(i+1))),
                    nn.LayerNorm(args.latent_dim//(2**(i+1))),
                    nn.LeakyReLU(),
                )
            )
        if self.proj_res_flag:
            self.proj_residual = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1))),out_features=(args.latent_dim//(2**(i+1)))*4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=(args.latent_dim//(2**(i+1)))*4,out_features=(args.latent_dim//(2**(i+1)))),
                )
                for i in range(args.proj_depth)
            ])
            self.proj_norm = nn.ModuleList([
                nn.LayerNorm(args.latent_dim//(2**(i+1)))
                for i in range(args.proj_depth)
            ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel, bias=False)

        in_dim = 2*args.latent_dim
        self.proj_norm_alpha = nn.LayerNorm(in_dim)
        self.out_alpha = nn.Linear(in_features=in_dim,out_features=args.in_channel, bias=False)

        self.proj_norm_beta = nn.LayerNorm(in_dim)
        self.out_beta = nn.Linear(in_features=in_dim,out_features=args.in_channel, bias=False)

    def _pre_process_input(self, inp):
        # print("inp", inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x", x.shape)
        h, w = x.shape[-2:]
        x = img2channel_list(x)
        # print("x", x.shape)
        inp = self._get_pre_activation(x, (h,w))
        return inp

    def _get_pre_activation(self, x, shape):
        # print('xa', x.shape)
        x = self.in_layer(x)
        # print('xb', x.shape)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
            # print('xc', x.shape)
        if self.pos_emb:
            x = x + img2channel_list(get_patch_position_embedding(x.size(-1), shape, device= x.device)).repeat(x.size(0),1,1)
        return x
    
    def _get_activation(self, x):
        x = self._concat_patch_central_pixel(x)
        x1 = torch.cat(x, dim=1)

        # print(x1.shape)
        k = self.proj_function(x1, value='k')
        # print(x.shape)

        # print(x1.shape)
        if self.output_activation=='control_sigmoid':
            a = self.proj_function(x1, value='alpha')
            b = self.proj_function(x1, value='beta')
            return self.controlled_sigmoid(k, a, b)
        elif self.output_activation=='exp':
            return torch.exp(k)
        elif self.output_activation=='relu':
            return torch.functional.F.relu(k)+1e-8
        elif self.output_activation=='sigmoid':
            return torch.functional.F.sigmoid(k)
        elif self.output_activation=='square':
            return torch.square(k)+1e-8
        else:
            raise ValueError(f'{self.output_activation} is not implimented.')
    

    def _concat_patch_central_pixel(self, inp):
        central_pixel = inp[:,inp.size(1)//2,:][:,None,:].expand_as(inp)
        return (torch.cat((inp,central_pixel), dim=-1), 
                torch.cat((central_pixel,inp), dim=-1))

    def proj_function(self, x, value='k'):
        if value=='k':
            for i in range(self.proj_depth):
                x = self.proj[i](x)
                if self.proj_res_flag:
                    x = x + self.proj_residual[i](x)
                    x = self.proj_norm[i](x)
            x = self.out(x)
            k1,k2 = torch.chunk(x, chunks=2, dim=1)
            x = (k1+k2)/(2*self.sigma)
        elif value=='alpha':
            x = self.proj_norm_alpha(x)
            x = self.out_alpha(x)
            a1,a2 = torch.chunk(x, chunks=2, dim=1)
            x = torch.functional.F.relu((a1+a2)/2)+1e-8
        elif value=='beta':
            x = self.proj_norm_beta(x)
            x = self.out_beta(x)
            b1,b2 = torch.chunk(x, chunks=2, dim=1)
            x = torch.functional.F.sigmoid((b1+b2)/2)
        else:
            raise ValueError(f'{value} is not implimented.')
        return x

    def controlled_sigmoid(self, x, a, b):
        # a(x) = Relu/sq(alpha) / ( 1 + exp ( - sigmoid (beta) x)
        return a/(1+torch.exp(-b*x))


    def _get_activation_img(self, x, pre=False):
        if pre:
            x = linear_img_fwd(x, self.in_layer)
            # print('xb', x.shape)
            for i in range(self.residual_depth):
                x = x + blk_img_fwd(x, self.in_residual[i])
                x = norm_img_fwd(x, self.in_norm[i])
                # print('xc', x)
            return x
        else:
            for i in range(self.proj_depth):
                x = blk_img_fwd(x, self.proj[i])
                if self.proj_res_flag:
                    x = x + blk_img_fwd(x, self.proj_residual[i])
                    x = norm_img_fwd(x, self.proj_norm[i])
            # print('xd', x.shape)
            x = linear_img_fwd(x, self.out)
            # print('xe', x.shape)
            return x

    def pre_feature(self,inp_img):
        inp_img = 2 * inp_img.clone() - 1
        x = self._extract_patch_vector(inp_img)
        x = self._get_activation_img(x, pre=True)
        return x

    def compute_activation(self,inp_img, t):
        x1, x2 = (torch.cat((inp_img, t), dim=1), torch.cat((t, inp_img), dim=1))
        x = torch.cat((x1,x2), dim=-1)
        # print('NLM-guide',x)
        x = self._get_activation_img(x)
        # print('NLM-out',x.shape)
        x1,x2 = torch.chunk(x, chunks=2, dim=-1)
        # print('NLM-out',x1.shape)
        # print('NLM-out',x2.shape)
        # return torch.square((x1+x2)/(-2*self.sigma)) + 1e-8
        return torch.exp((x1+x2)/(-2*self.sigma))

    def __str__(self):
        return 'c_v5'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

class NKD_diff_kernel_learn(NKD_denoisers):
    def __init__(self, args, device): 
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel

        if args.blind:
            in_dim=args.in_channel*patch_size*patch_size
        else:
            in_dim=(args.in_channel+1)*patch_size*patch_size

        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)
        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*2),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*2,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])
        # self.in_proj = nn.Linear(in_features=in_dim,out_features=latent_dim)

        self.proj = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim//(2**i),out_features=args.latent_dim//(2**(i+1))),
                nn.LayerNorm(args.latent_dim//(2**(i+1))),
                nn.LeakyReLU(),
            )
            for i in range(args.proj_depth)
        ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel)

    def _diff_sq_central_pixel(self, inp):
        b, l, c = inp.shape
        central_pixel = repeat(inp[:,l//2,:], 'b c -> b l c', l=l)
        return (inp-central_pixel)**2

    def _get_activation(self, inp):
        x1 = self.in_layer(inp)
        for i in range(self.residual_depth):
            x1 = x1 + self.in_residual[i](x1)
            x1 = self.in_norm[i](x1)
        # x = self.in_proj(x)

        # print('x2', x1)
        x2 = self._diff_sq_central_pixel(x1)
        # print('x3', x2)
        x3 = self.proj(x2)
        x4 = self.out(x3)
        return torch.exp(x4/(-2))

    def _get_activation_img(self, inp):
        x1 = linear_img_fwd(inp, self.in_layer)
        for i in range(self.residual_depth):
            x1 = x1 + blk_img_fwd(x1, self.in_residual[i])
            x1 = norm_img_fwd(x1,self.in_norm[i])
        # x = self.in_proj(x)

        # print('x2', x1)
        x2 = self._diff_sq_central_pixel(x1)
        # print('x3', x2)
        x3 = blk_img_fwd(x2, self.proj)
        x4 = linear_img_fwd(x3, self.out)
        return torch.exp(x4/(-2))

    def _pre_process_input(self, inp):
        # print('inp', inp)
        x = self._extract_patch_vector(inp)
        # print('x', x)
        x = img2channel_list(x)
        # print('x', x)
        return x

    def __str__(self):
        return 'diff_kl'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

class NKD_rbf_feat_learn(NKD_denoisers):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.residual_depth = args.residual_depth
        self.proj_depth = args.proj_depth
        self.sigma = 1
        self.blind = args.blind
        self.latent_dim = args.latent_dim
        self.latent_channel_dim = args.latent_channel_dim
        self.patch_rad = args.patch_rad
        patch_size = args.patch_rad * 2 + 1

        self.window_rad = args.window_rad
        self.in_channel = args.in_channel

        if args.blind:
            in_dim=args.in_channel*patch_size*patch_size
        else:
            in_dim=(args.in_channel+1)*patch_size*patch_size

        self.in_dim=in_dim
        self.in_layer = nn.Linear(in_features=in_dim,out_features=args.latent_dim)
        self.in_residual = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim,out_features=args.latent_dim*2),
                nn.LeakyReLU(),
                nn.Linear(in_features=args.latent_dim*2,out_features=args.latent_dim),
            )
            for _ in range(args.residual_depth)
        ])
        self.in_norm = nn.ModuleList([
            nn.LayerNorm(args.latent_dim)
            for _ in range(args.residual_depth)
        ])
        # self.in_proj = nn.Linear(in_features=in_dim,out_features=latent_dim)

        self.proj = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=args.latent_dim//(2**i),out_features=args.latent_dim//(2**(i+1))),
                nn.LayerNorm(args.latent_dim//(2**(i+1))),
                nn.LeakyReLU(),
            )
            for i in range(args.proj_depth)
        ])
        self.out = nn.Linear(in_features=args.latent_dim//(2**(args.proj_depth)),out_features=args.in_channel*self.latent_channel_dim)

    def _diff_sq_patch_central_pixel(self, inp):
        b, n, c = inp.shape
        central_pixel = (inp[:,n//2,:])[:,None,:].expand_as(inp)
        return (inp-central_pixel)**2

    def _get_features(self, x):
        x = self.in_layer(x)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
        # x = self.in_proj(x)

        # print('x1', x.shape)
        x = self.proj(x)

        # print('x2', x.shape)
        x = self.out(x)
        return x

    def _get_activation(self, x):
        x = self._get_features(x)

        # print('x6', x.shape)
        out = self._diff_sq_patch_central_pixel(x)
        # print('out', out.shape)

        b, c, d = out.shape
        out = torch.sum(out.reshape(b, c, self.latent_channel_dim, self.in_channel), dim=-2)
        # print('out', out.shape)
        return torch.exp((out)/(-2))

    def _get_activation_img(self, x):
        x = linear_img_fwd(x, self.in_layer)
        for i in range(self.residual_depth):
            x = x + blk_img_fwd(x, self.in_residual[i])
            x = norm_img_fwd(x, self.in_norm[i])
        # x = self.in_proj(x)

        # print('x1', x.shape)
        x = blk_img_fwd(x, self.proj)

        # print('x2', x.shape)
        x = linear_img_fwd(x, self.out)
        
        # print('x6', x.shape)
        out = self._diff_sq_patch_central_pixel(x)
        # print('out', out.shape)

        b, c, d = out.shape
        out = torch.sum(out.reshape(b, c, self.latent_channel_dim, self.in_channel), dim=-2)
        # print('out', out.shape)
        return torch.exp((out)/(-2))

    def _pre_process_input(self, inp):
        # print('process_inp: o-_extract_patch_vector>',x)
        x = self._extract_patch_vector(inp)
        # print('process_inp: _extract_patch_vector output>',x)
        return img2channel_list(x)

    def __str__(self):
        return 'rbf'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}{self.latent_channel_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

    # def apply2image(self, inp):


architectures = {
    'NKD_concat_kernel_learn': NKD_concat_kernel_learn,
    'NKD_concat_kernel_learn_v2': NKD_concat_kernel_learn_v2,
    'NKD_concat_kernel_learn_v3': NKD_concat_kernel_learn_v3,
    'NKD_concat_kernel_learn_v4': NKD_concat_kernel_learn_v4,
    'NKD_concat_kernel_learn_v5': NKD_concat_kernel_learn_v5,
    'NKD_diff_kernel_learn': NKD_diff_kernel_learn,
    'NKD_rbf_feat_learn': NKD_rbf_feat_learn,
    'KPN': KPN,
}
