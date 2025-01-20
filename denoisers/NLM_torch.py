
## imports
import numpy as np
import torch

######################################################################################
######################################################################################
#------------------              DSG NLM FAST              ----------------------------



def integral_img_sq_diff(v,dx,dy):
    t = img2Dshift(v,dx,dy)
    diff = (v-t)**2
    #diff to double
    diff = diff.to(torch.float32)
    sd = torch.cumsum(diff,dim=-2)
    sd = torch.cumsum(sd,dim=-1)
    return(sd,diff,t)


def triangle(dx,dy,Ns):
    # r1 = np.abs(1 - np.abs(dx)/(Ns+1))
    #r1 is the positive difference between 1 and the absolute value of dx divided by Ns+1
    r1 = max(1 - (abs(dx)/(Ns+1)), 0)
    # r2 = np.abs(1 - np.abs(dy)/(Ns+1))
    #r2 is the positive difference between 1 and the absolute value of dy divided by Ns+1
    r2 = max(1 - (abs(dy)/(Ns+1)), 0)
    return r1*r2


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


#thus the seperable trainagle filter takes in a 2D array and returns a 2D array
#the inputs are: dx, dy, Ns
#the output is: Hat
def separable_triangle(dx, dy, Ns, height, width):

    # Calculate the value from dx, dy, and Ns
    value = max(0, 1 - (abs(dx) / (Ns + 1))) * max(0, 1 - (abs(dy) / (Ns + 1)))

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


#we write the main function to compute the DS_NLM algorithm
#we will implement the algorithm DSG_NLm written above of 4 steps
#we have 3 loops in the algorithm
#this algorithm is kind of loop unrolling

def DSG_NLM(input_image,guide_image,patch_rad,window_rad,sigma):
    #convert input_image to double
    input_image = input_image.to(torch.float32)
    guide_image = guide_image.to(torch.float32)

    b,channel,height,width = input_image.shape
    u = torch.zeros((b, channel, height, width)).to(torch.float64)
    
    u = torch.zeros((b, channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold denoised image
    W0 = torch.zeros((b, channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold accumulated weights
    
    padded_guide = torch.nn.functional.pad(guide_image, (patch_rad, patch_rad, patch_rad, patch_rad), mode='circular')
    padded_v = torch.nn.functional.pad(input_image, (window_rad, window_rad, window_rad, window_rad), mode='circular')
    normalization_factor = -(sigma*sigma)

    # 0th loop
    W0 = torch.zeros((b,channel,height,width)).to(device=input_image.device, dtype=torch.float64)
    for dx in range(-window_rad,window_rad+1):
        for dy in range(-window_rad,window_rad+1):
            sd,_,_ = integral_img_sq_diff(padded_guide,dx,dy)
            hat = separable_triangle(dx,dy,window_rad,height,width).to(input_image.device)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[:, :, patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * torch.exp(sqdist1/normalization_factor).to(torch.float64)

            W0 = W0 + w

    # 1st loop
    W1 = torch.zeros((b,channel,height,width)).to(device=input_image.device, dtype=torch.float64)
    for dx in range(-window_rad,window_rad+1):
        for dy in range(-window_rad,window_rad+1):
            sd,_,_ = integral_img_sq_diff(padded_guide,dx,dy)
            hat = separable_triangle(dx,dy,window_rad,height,width).to(input_image.device)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[:, :, patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * torch.exp(sqdist1/normalization_factor).to(torch.float64)
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
                sd,_,_ = integral_img_sq_diff(padded_guide,dx,dy)
                hat = separable_triangle(dx,dy,window_rad,height,width).to(input_image.device)
                temp1 = img2Dshift(sd,patch_rad,patch_rad)
                temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
                temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
                temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
                res = temp1 + temp2 - temp3 - temp4
                sqdist1 = res[:, :, patch_rad:patch_rad+height,patch_rad:patch_rad+width]
                w = hat * torch.exp(sqdist1/normalization_factor).to(torch.float64)
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

######################################################################################
######################################################################################
#------------------              DSG NLM FAST ENDS         ----------------------------


def NLM_tensor(input_image,guide_image,patch_rad,window_rad,sigma, return_D_matrix=False):
    #convert input_image to double
    input_image = input_image.to(torch.float32)
    guide_image = guide_image.to(torch.float32)

    _, channel,height,width = input_image.shape

    U = torch.zeros((channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold denoised image
    Z = torch.zeros((channel, height, width), dtype=torch.float32, device=input_image.device)  # To hold accumulated weights

    padded_guide = torch.nn.functional.pad(guide_image, (patch_rad, patch_rad, patch_rad, patch_rad), mode='circular')
    padded_v = torch.nn.functional.pad(input_image, (window_rad, window_rad, window_rad, window_rad), mode='circular')

    # normalization_factor = (2*patch_rad*patch_rad*sigma*sigma)
    normalization_factor = -(sigma*sigma)

    for dx in range(-window_rad, window_rad + 1):
        for dy in range(-window_rad, window_rad + 1):
            sd,_,_ = integral_img_sq_diff(padded_guide,dx,dy)
            hat = separable_triangle(dx,dy,window_rad,height,width).to(input_image.device)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[:, :, patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            # print(hat.shape, sqdist1.shape)
            w = hat * torch.exp(sqdist1/normalization_factor).to(torch.float64)
            v = padded_v[:, :, window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
            U = U + w*v
            Z = Z + w

    U = U/Z
    if return_D_matrix:
        return U , Z    # U = Denoised image, Z = Normalization coefficients
    return U.to(torch.float32)        # U = Denoised image



