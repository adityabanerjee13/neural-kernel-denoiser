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

dtype = torch.float32
#nlm_op
# Inherit from Function
class NLM_Operator(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input_image, guide_features, window_rad, sigma):
        input_image = input_image.to(torch.float32)
        guide = guide_features.to(torch.float32)

        req_guide_grad = ctx.needs_input_grad[1]
        req_input_grad = ctx.needs_input_grad[0]

        _, _, height, width = input_image.shape
        _, _, l, height, width = guide.shape

        U = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold denoised image
        Z = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold accumulated weights
        if req_guide_grad:
            dZ = torch.zeros_like(guide_features, dtype=torch.float64, device=input_image.device)  # To hold accumulated weights

        padded_v = torch.nn.functional.pad(input_image, (window_rad, window_rad, window_rad, window_rad), mode='circular')

        normalization_factor = -(sigma * sigma)

        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                t = img2Dshift(guide, dx, dy)
                diff = (guide - t)
                sqdist = torch.sum(diff**2, dim=2)
                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)

                w = hat * torch.exp(sqdist/normalization_factor)
                v = padded_v[:, :, window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                U = U + w * v
                Z = Z + w
                # print(diff.shape, w.shape, guide_features.shape)
                if req_guide_grad:
                    dZ = dZ + diff * repeat(w, 'b c h w -> b c l h w', l=l)

        U = U / Z

        if req_guide_grad:
            ctx.save_for_backward(input_image, guide, U, Z, dZ, torch.tensor([window_rad], device=input_image.device), torch.tensor([sigma], device=input_image.device))

        return U.to(torch.float32)  # U = Denoised image

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward 
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to 
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input_image, guide, U, Z, dZ, window_rad, sigma = ctx.saved_tensors
        grad_input = None

        _, _, height, width = input_image.shape
        _, _, l, height, width = guide.shape

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        NdLo = grad_output/Z
        grad_guide = torch.zeros_like(guide, dtype=torch.float64, device=input_image.device)  # To hold denoised image

        normalization_factor = -(sigma.item() * sigma.item())

        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                if dx==0 and dy==0:
                    grad_guide = grad_guide - repeat(NdLo * U, 'b c h w -> b c l h w', l=l) * dZ
                    continue
                guide_kl = img2Dshift(guide, dx, dy)
                diff = (guide - guide_kl)
                sqdist = torch.sum(diff**2, dim=2)
                exp = torch.exp(sqdist/normalization_factor)
                input_kl = img2Dshift(input_image, dx, dy)
                NdLo_kl = img2Dshift(NdLo, dx, dy)
                U_kl = img2Dshift(U, dx, dy)
                temp = NdLo_kl * (input_image - U_kl) + NdLo * input_kl
                mask = Box(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)

                grad_guide = grad_guide + mask * (diff * repeat(exp * temp, 'b c h w -> b c l h w', l=l))
        grad_guide = grad_guide * (2 / normalization_factor)
        return grad_input, grad_guide, None, None

# def NLM(input_image, guide_features, window_rad, sigma):
#     return NLM_Operator.apply(input_image, guide_features, window_rad, sigma)

def NLM(input_image, guide_features, window_rad, sigma):
        input_image = input_image.to(torch.float32)
        guide = guide_features.to(torch.float32)

        _, _, height, width = input_image.shape

        U = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold denoised image
        Z = torch.zeros_like(input_image, dtype=torch.float64, device=input_image.device)  # To hold accumulated weights

        padded_v = torch.nn.functional.pad(input_image, (window_rad, window_rad, window_rad, window_rad), mode='circular')

        normalization_factor = -(sigma * sigma)

        for dx in range(-window_rad, window_rad + 1):
            for dy in range(-window_rad, window_rad + 1):
                t = img2Dshift(guide, dx, dy)
                diff = (guide - t)
                sqdist = torch.sum(diff**2, dim=2)
                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)

                w = hat * torch.exp(sqdist/normalization_factor)
                v = padded_v[:, :, window_rad + dx:window_rad + dx + height, window_rad + dy:window_rad + dy + width]
                U = U + w * v
                Z = Z + w

        U = U / Z

        return U.to(torch.float32)  # U = Denoised image

class DSG_NLM_Operator(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input_image, guide_image, window_rad, sigma):
        #convert input_image to double
        input_image = input_image.to(torch.float32)
        guide = guide_image.to(torch.float32)

        b,channel,height,width = input_image.shape
        u = torch.zeros((b, channel, height, width)).to(torch.float64)
        
        u = torch.zeros((b, channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold denoised image
        W0 = torch.zeros((b, channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold accumulated weights

        # guide = torch.nn.functional.pad(guide_image, (patch_rad, patch_rad, patch_rad, patch_rad), mode='circular')
        padded_v = torch.nn.functional.pad(input_image, (window_rad, window_rad, window_rad, window_rad), mode='circular')

        normalization_factor = -(sigma*sigma)

        # 0th loop
        W0 = torch.zeros((b,channel,height,width)).to(device=input_image.device, dtype=torch.float64)
        for dx in range(-window_rad,window_rad+1):
            for dy in range(-window_rad,window_rad+1):
                t = img2Dshift(guide, dx, dy)
                diff = (guide - t)
                sqdist = torch.sum(diff**2, dim=2)
                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)
                w = hat * torch.exp(sqdist/normalization_factor)

                W0 = W0 + w

        # 1st loop
        W1 = torch.zeros((b,channel,height,width)).to(device=input_image.device, dtype=torch.float64)
        for dx in range(-window_rad,window_rad+1):
            for dy in range(-window_rad,window_rad+1):
                t = img2Dshift(guide, dx, dy)
                diff = (guide - t)
                sqdist = torch.sum(diff**2, dim=2)
                hat = separable_triangle(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)
                w = hat * torch.exp(sqdist/normalization_factor)

                W0_pad = torch.nn.functional.pad(W0,(window_rad, window_rad, window_rad, window_rad),mode='circular')
                W0_shift = img2Dshift(W0_pad,dx,dy)
                W0_temp = W0_shift[:, :, window_rad:window_rad+height,window_rad:window_rad+width]
                w1 = w / (torch.sqrt(W0)*torch.sqrt(W0_temp)).to(torch.float64)
                W1 = W1 + w1
        
        # 2nd loop
        alpha = 1/torch.max(W1).to(torch.float64)
        W2 = torch.zeros((b,channel,height,width)).to(device=input_image.device, dtype=torch.float64)
        for dx in range(-window_rad,window_rad+1):
            for dy in range(-window_rad,window_rad+1):
                if((dx != 0) or (dy != 0)):
                    t = img2Dshift(guide, dx, dy)
                    diff = (guide - t)
                    sqdist = torch.sum(diff**2, dim=2)
                    hat = separable_triangle(dx, dy, window_rad, height, width).to(device=input_image.device, dtype = torch.float64)
                    w = hat * torch.exp(sqdist/normalization_factor)

                    W0_pad = torch.nn.functional.pad(W0,(window_rad, window_rad, window_rad, window_rad),mode='circular')
                    W0_shift = img2Dshift(W0_pad,dx,dy)
                    W0_temp = W0_shift[:, :, window_rad:window_rad+height,window_rad:window_rad+width]
                    w2 = (alpha*w) / (torch.sqrt(W0)*torch.sqrt(W0_temp)).to(torch.float64)
                    v = padded_v[:, :, window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
                    u = u + w2*v
                    W2 = W2 + w2

        u = u + (1-W2)*input_image
        #convert u to double
        u = u.to(torch.float64)
        return u        # u = Denoised image

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError('Back-propagation through DSG-NLM has not been implemented')

def DSG_NLM(input_image, guide_features, window_rad, sigma):
    return DSG_NLM_Operator.apply(input_image, guide_features, window_rad, sigma)



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


class NKD_latent_layers(nn.Module):
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
            self.in_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )
        else:
            self.in_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel+1,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                nn.LeakyReLU()
            )

        self.mid_block = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim//2,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(latent_dim//2, latent_dim, kernel_size=1, stride=1, padding=0),
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.down1 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim//2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.norm2 = nn.LayerNorm(latent_dim//2)
        self.down2 = nn.Sequential(
            nn.Conv2d(latent_dim//2, latent_dim//4, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.norm3 = nn.LayerNorm(latent_dim//4)
        self.down3 = nn.Conv2d(latent_dim//4, latent_channel_dim*in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, inp, sig=None):
        print('f',inp.shape)
        guide = self.get_features(inp, sig)
        print('g',guide.shape)
        x = NLM(inp, guide, self.window_rad, self.sigma)
        return x
    
    def dsg_output(self, inp, sig=None):
        guide = self.get_features(inp, sig)
        x = DSG_NLM(inp, guide, self.window_rad, self.sigma)
        return x

    def get_features(self, guide, sig=None):
        b, _, h, w = guide.shape
        if self.blind:
            temp = torch.nn.functional.pad(guide, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
            x = self.in_block(temp)
        else:
            assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
            temp = torch.cat(
                (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                dim = 1
            )
            temp = torch.nn.functional.pad(temp, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
            x = self.in_block(temp)

        res = x
        x = self.mid_block(x)
        x = x + res

        x = self.norm1(x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()
        x = self.down1(x)
        x = self.norm2(x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()
        x = self.down2(x)
        x = self.norm3(x.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()
        x = self.down3(x)
        x = rearrange(x, 'b (c l) h w -> b c l h w', l=self.l)

        return x

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


class NKD_latent_layers_v2(nn.Module):
    def __init__(self,patch_rad,window_rad,in_channel,latent_dim,latent_channel_dim,device,blind=False,denoiser='NLM'):
        super().__init__()
        self.device = device
        n_heads=4
        self.n_heads = n_heads
        self.blind = blind
        self.L = latent_dim
        self.l = latent_channel_dim
        self.sigma = 1
        self.patch_rad = patch_rad
        patch_size = patch_rad * 2 + 1

        self.window_rad = window_rad
        self.in_channel = in_channel

        self.in_block = nn.ModuleList([])
        self.mid_block = nn.ModuleList([])
        self.norm1 = nn.ModuleList([])
        self.down1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        self.down2 = nn.ModuleList([])
        self.norm3 = nn.ModuleList([])
        self.down3 = nn.ModuleList([])
        for _ in range(in_channel):
            if blind:
                self.in_block.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                    nn.LeakyReLU()
                ))
            else:
                self.in_block.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_channel+1,out_channels=latent_dim,kernel_size=patch_size,stride=1,padding=0),
                    nn.LeakyReLU()
                ))
            self.mid_block.append(nn.Sequential(
                nn.Conv2d(latent_dim,latent_dim//2,kernel_size=1,stride=1,padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(latent_dim//2, latent_dim, kernel_size=1, stride=1, padding=0),
            ))
            self.norm1.append(nn.BatchNorm2d(latent_dim))
            self.down1.append(nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim//2, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),
            ))
            self.norm2.append(nn.BatchNorm2d(latent_dim//2))
            self.down2.append(nn.Sequential(
                nn.Conv2d(latent_dim//2, latent_dim//4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(),
            ))
            self.norm3.append(nn.BatchNorm2d(latent_dim//4))
            self.down3.append(nn.Conv2d(latent_dim//4, latent_channel_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, inp, sig=None):
        guide = self.get_features(inp, sig)
        x = NLM(inp, guide, self.window_rad, self.sigma)
        return x

    def get_features(self, guide, sig=None):
        b, _, h, w = guide.shape
        # pos_emb = get_patch_position_embedding(self.L, (h, w), self.device)
        for i in range(self.in_channel):
            if self.blind:
                temp = torch.nn.functional.pad(guide, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
                x = self.in_block[i](temp)
            else:
                assert sig!=None, 'In the case of NON-blind model batch std. must be an input'
                temp = torch.cat(
                    (guide, torch.ones((b,1,h,w), device=self.device, dtype=guide.dtype)*sig),
                    dim=1,
                )
                temp = torch.nn.functional.pad(temp, (self.patch_rad, self.patch_rad, self.patch_rad, self.patch_rad), mode='circular')
                x = self.in_block[i](temp)

            # x = x + pos_emb

            res = x
            x = self.mid_block[i](x)
            x = x + res

            x = self.norm1[i](x)
            x = self.down1[i](x)
            x = self.norm2[i](x)
            x = self.down2[i](x)
            x = self.norm3[i](x)
            x = self.down3[i](x)
            if i==0:
                feat = x.unsqueeze(1)
            else:
                feat = torch.cat((feat, x.unsqueeze(1)), dim=1)
        return feat

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

