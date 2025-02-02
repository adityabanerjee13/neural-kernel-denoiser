import torch
import torch.nn as nn
import os
from einops import repeat
from utils import *



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
        t = torch.zeros(channel*patch_size*patch_size, channel, patch_size, patch_size, device=inp.device)
        count=0
        for c in range(channel):
            for i in range(patch_size):
                for j in range(patch_size):
                    t[count,c,i,j] = 1
                    count+=1

        return torch.functional.F.conv2d(inp, t)#.reshape(batch, -1, channel*patch_s*patch_s)

    # Defined in child class
    def _process_input(self):
        # to be defined in the child class
        pass

    # Defined in child class
    def _get_activation(self):
        # to be defined in the child class
        pass

    def forward(self, inp, sig=None):
        # print(torch.max(inp))
        # print(torch.min(inp))
        b, c, h, w = inp.shape

        guide = 2 * inp.clone() - 1
        b, _, h, w = guide.shape
        if self.blind:
            pass
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            guide = torch.cat(
                (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                dim = 1
            )

        # Extract guide weights
        # print("guide",guide.shape)
        guide = self._process_input(guide)
        # print("guide",guide.shape)
        guide_weights = self._get_activation(guide)
        # print("guide_weights",guide_weights)
        normalizing_factor = torch.sum(guide_weights, dim=1)
        # print("normalizing_factor",normalizing_factor)

        inp = inp[:, :, self.patch_rad:h-self.patch_rad, self.patch_rad:w-self.patch_rad].reshape(b,-1,self.in_channel)
        # print("inp",inp)
        out = torch.sum((inp*guide_weights), dim=1)/normalizing_factor
        # print("out",out)
        return out.reshape(b, self.in_channel)

    def forward_img(self, img, sig=None):
        # Extract patches from image window
        print(img.shape)
        img_patches = extract_all_patches(img, 2*(self.window_rad+self.patch_rad)+1)
        print(img_patches.shape)
        return self(img_patches)
        
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
                    clean= clean.reshape(-1,c)

            # More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            output = self(noisy, sigma)

            if lamda!=0:
                l1 = l1_loss(self)
            else:
                l1 = 0

            loss = criterion(output, clean)
            loss_t = loss + lamda*l1
            loss_t.backward()
            optimizer.step()

            b = output.size(0)
            count+=b
            avg_loss += (loss.item() * b)

        avg_loss = avg_loss/count
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
            checkpoint_path = os.path.join(out_folder, f'model{_state}_best.pth')
        else:
            checkpoint_path = os.path.join(out_folder, f'model{_state}_step_{step}.pth')
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

    def _concat_patch_central_pixel(self, inp):
        b, c, h, w = inp.shape
        central_pixel = (inp[:,:,h//2,w//2])[:,:,None,None].expand_as(inp).permute(0,2,3,1).reshape(b,-1,c)
        return (torch.cat((inp.permute(0,2,3,1).reshape(b,-1,c),central_pixel), dim=-1), 
                torch.cat((central_pixel,inp.permute(0,2,3,1).reshape(b,-1,c)), dim=-1))

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
        return torch.exp((x1+x2)/(-2))

    def _process_input(self, inp):
        # print("inp",inp.shape)
        x = self._extract_patch_vector(inp)
        # print("x",x.shape)
        x1, x2 = self._concat_patch_central_pixel(x)
        # print("x1",x1.shape)
        # print("x2",x2.shape)
        inp = torch.cat((x1,x2), dim=1)
        # print("inp",inp.shape)
        return inp

    def __str__(self):
        return '_concat_kl'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

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

    def _process_input(self, inp):
        # print('inp', inp)
        x = self._extract_patch_vector(inp)
        b, l, _, _ = x.shape
        # print('x', x)
        x = x.permute(0,2,3,1).reshape(b,-1,l)
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

    def _get_activation(self, x):
        x = self.in_layer(x)
        for i in range(self.residual_depth):
            x = x + self.in_residual[i](x)
            x = self.in_norm[i](x)
        # x = self.in_proj(x)

        # print('x1', x.shape)
        x = self.proj(x)

        # print('x2', x.shape)
        x = self.out(x)
        
        # print('x6', x.shape)
        out = self._diff_sq_patch_central_pixel(x)
        # print('out', out.shape)

        b, c, d = out.shape
        out = torch.sum(out.reshape(b, c, self.latent_channel_dim, self.in_channel), dim=-2)
        # print('out', out.shape)
        return torch.exp((out)/(-2))

    def _process_input(self, inp):
        b, c, _, _ = inp.shape
        # print('process_inp: o-_extract_patch_vector>',x)
        x = self._extract_patch_vector(inp).permute(0,2,3,1).reshape(b, -1, self.in_dim)
        # print('process_inp: _extract_patch_vector output>',x)
        return x

    def __str__(self):
        return 'rbf'+f'_{self.patch_rad}{self.window_rad}{self.in_channel}{self.latent_dim}{self.latent_channel_dim}_{self.residual_depth}{self.proj_depth}_{int(self.blind)}'

    # def apply2image(self, inp):

architectures = {
    'NKD_concat_kernel_learn': NKD_concat_kernel_learn,
    'NKD_diff_kernel_learn': NKD_diff_kernel_learn,
    'NKD_rbf_feat_learn': NKD_rbf_feat_learn,
}
