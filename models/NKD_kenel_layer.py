import torch
import torch.nn as nn
from einops import rearrange, repeat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import *



# Overheads
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

class NKD_kernel_layers_v1(nn.Module):
    def __init__(self, patch_rad, window_rad, in_channel, latent_dim, latent_channel_dim, device, blind=False, denoiser='NLM'):
        super().__init__()
        self.device = device
        n_heads=4
        self.n_heads = n_heads
        self.sigma = 1
        self.blind = blind
        self.latent_dim = latent_dim
        self.l = latent_channel_dim
        self.patch_rad = patch_rad
        patch_size = patch_rad * 2 + 1

        self.window_rad = window_rad
        self.in_channel = in_channel

        if blind:
            self.in1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )
        else:
            self.in1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel+1,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )

        self.mid1 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim*2,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim*2,out_channels=latent_dim,kernel_size=1,stride=1,padding=0),
        )

        self.asymm_merg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim//2,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim//2,out_channels=in_channel,kernel_size=1,stride=1,padding=0),
        )

    def forward(self, inp, sig=None):
        guide = inp.clone()
        # print(guide.shape)
        b, _, h, w = guide.shape
        if self.blind:
            temp = torch.nn.functional.pad(guide, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            temp = torch.cat(
                (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                dim = 1
            )
            temp = torch.nn.functional.pad(temp, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')

        l1 = self.mid1(self.in1(temp))

        _, _, height, width = inp.shape
        window_rad = self.window_rad
        U = torch.zeros_like(inp, dtype=torch.float64, device=inp.device)  # To hold denoised image
        Z = torch.zeros_like(inp, dtype=torch.float64, device=inp.device)  # To hold accumulated weights

        padded_v = torch.nn.functional.pad(inp, (window_rad, window_rad, window_rad, window_rad), mode='circular')

        normalization_factor = -(self.sigma * self.sigma)

        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                s_l1 = img2Dshift(l1, dx, dy)
                # print(s_l1.shape, s_l2.shape, torch.cat((l1, s_l2), dim=1).shape)

                sim = self.asymm_merg_block1((l1 - s_l1)**2)

                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=inp.device, dtype = torch.float64)
                w = hat * torch.exp(sim/normalization_factor)

                v = padded_v[:, :, window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                U = U + w * v
                Z = Z + w

        U = U / Z

        return U.to(torch.float32)  # U = Denoised image

    def train_step(self, train_loader, criterion, optimizer, lamda = 1e-3):
        self.train()
        loss = 0

        for clean, noisy, sigma in train_loader:
            clean = clean.to(self.device, dtype=torch.float32)
            noisy = noisy.to(self.device, dtype=torch.float32)
            sigma = sigma.to(self.device, dtype=torch.float32)
            b = clean.size(0)

            # More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            if self.blind:
                output = self(noisy)
            else:
                output = self(noisy, sigma)

            if lamda!=0:
                l1 = l1_loss(self)
            else:
                l1 = 0
            loss = criterion(output, clean)
            loss_t = loss + lamda*l1
            loss_t.backward()
            optimizer.step()

            loss = (loss.item() * b)

        loss = loss/len(train_loader.dataset)
        return loss

    def val_step(self, val_loader):
        self.eval()
        metric = {'psnr' : 0, 'ssim' : 0}

        with torch.no_grad():
            for clean, noisy, sigma in val_loader:
                clean = clean.to(self.device, dtype=torch.float32)
                noisy = noisy.to(self.device, dtype=torch.float32)
                sigma = sigma.to(self.device, dtype=torch.float32)
                b = clean.size(0)

                if self.blind:
                    output = self(noisy)
                else:
                    output = self(noisy, sigma)

                metric['psnr'] += calculate_psnr(output, clean)
                metric['ssim'] += calculate_ssim(output, clean)

        metric['ssim'] = metric['ssim']/len(val_loader)
        metric['psnr'] = metric['psnr']/len(val_loader)
        return metric




class NKD_kernel_layers(nn.Module):
    def __init__(self, patch_rad, window_rad, in_channel, latent_dim, latent_channel_dim, device, blind=False, denoiser='NLM'):
        super().__init__()
        self.device = device
        n_heads=4
        self.n_heads = n_heads
        self.sigma = 1
        self.blind = blind
        self.latent_dim = latent_dim
        self.l = latent_channel_dim
        self.patch_rad = patch_rad
        patch_size = patch_rad * 2 + 1

        self.window_rad = window_rad
        self.in_channel = in_channel

        if blind:
            self.in1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )
            self.in2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )
        else:
            self.in1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel+1,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )
            self.in2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channel+1,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )

        self.mid1 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim//4,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim//4,out_channels=latent_dim//2,kernel_size=1,stride=1,padding=0),
        )

        self.mid2 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim//4,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim//4,out_channels=latent_dim//2,kernel_size=1,stride=1,padding=0),
        )

        self.asymm_merg_block1 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim//2,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim//2,out_channels=in_channel,kernel_size=1,stride=1,padding=0),
        )
        self.asymm_merg_block2 = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim,out_channels=latent_dim//2,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=latent_dim//2,out_channels=in_channel,kernel_size=1,stride=1,padding=0),
        )

        self.asymm_merg_block2[0].weight = self.asymm_merg_block1[0].weight
        self.asymm_merg_block2[2].weight = self.asymm_merg_block1[2].weight


    def forward(self, inp, sig=None):
        guide = inp.clone()
        # print(guide.shape)
        b, _, h, w = guide.shape
        if self.blind:
            temp = torch.nn.functional.pad(guide, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            temp = torch.cat(
                (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                dim = 1
            )
            temp = torch.nn.functional.pad(temp, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
        # print(temp.shape)
        l1 = self.mid1(self.in1(temp))
        # print(l1.shape)
        l2 = self.mid2(self.in2(temp))

        _, _, height, width = inp.shape
        window_rad = self.window_rad
        U = torch.zeros_like(inp, dtype=torch.float64, device=inp.device)  # To hold denoised image
        Z = torch.zeros_like(inp, dtype=torch.float64, device=inp.device)  # To hold accumulated weights

        padded_v = torch.nn.functional.pad(inp, (window_rad, window_rad, window_rad, window_rad), mode='circular')

        normalization_factor = -(self.sigma * self.sigma)

        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                s_l1 = img2Dshift(l1, dx, dy)
                s_l2 = img2Dshift(l2, dx, dy)
                # print(s_l1.shape, s_l2.shape, torch.cat((l1, s_l2), dim=1).shape)

                a_ij = self.asymm_merg_block1(torch.cat((l1, s_l2), dim=1))
                a_ji = self.asymm_merg_block2(torch.cat((l2, s_l1), dim=1))

                sim = (a_ij + a_ji)/2

                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=inp.device, dtype = torch.float64)
                w = hat * torch.exp(sim/normalization_factor)

                v = padded_v[:, :, window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                U = U + w * v
                Z = Z + w

        U = U / Z

        return U.to(torch.float32)  # U = Denoised image

    def train_step(self, train_loader, criterion, optimizer, lamda = 1e-3):
        self.train()
        loss = 0

        for clean, noisy, sigma in train_loader:
            clean = clean.to(self.device, dtype=torch.float32)
            noisy = noisy.to(self.device, dtype=torch.float32)
            sigma = sigma.to(self.device, dtype=torch.float32)
            b = clean.size(0)

            # More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            if self.blind:
                output = self(noisy)
            else:
                output = self(noisy, sigma)

            if lamda!=0:
                l1 = l1_loss(self)
            else:
                l1 = 0
            loss = criterion(output, clean)
            loss_t = loss + lamda*l1
            loss_t.backward()
            optimizer.step()

            loss = (loss.item() * b)

        loss = loss/len(train_loader.dataset)
        return loss

    def val_step(self, val_loader):
        self.eval()
        metric = {'psnr' : 0, 'ssim' : 0}

        with torch.no_grad():
            for clean, noisy, sigma in val_loader:
                clean = clean.to(self.device, dtype=torch.float32)
                noisy = noisy.to(self.device, dtype=torch.float32)
                sigma = sigma.to(self.device, dtype=torch.float32)
                b = clean.size(0)

                if self.blind:
                    output = self(noisy)
                else:
                    output = self(noisy, sigma)

                metric['psnr'] += calculate_psnr(output, clean)
                metric['ssim'] += calculate_ssim(output, clean)

        metric['ssim'] = metric['ssim']/len(val_loader)
        metric['psnr'] = metric['psnr']/len(val_loader)
        return metric


