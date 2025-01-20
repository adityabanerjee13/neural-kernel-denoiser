import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from PnP.linear_models import linear


def Inf_median_filter_2D(image, kernal_radius=5):
    #kernal radius should be an odd number, if not we will make it odd by decrementing it by 1,
    #if kernal radius is 0, assert error
    assert kernal_radius > 0, 'kernal radius should be greater than 0'
    if kernal_radius % 2 == 0:
        kernal_radius = kernal_radius - 1
    #convert data type to float64
    image = image.astype(np.float64)
    #we will pad the image on margin on all 4 sides with Inf
    padded_image = np.pad(image, kernal_radius, 'constant', constant_values=np.inf)

    #we will get the indices in the image inpainted_image where value is Inf
    indices_infinity = np.where(np.isinf(image))
    #we will change the pixel values in the padded image and replace the Inf with the median of the patch
    #the we will extract the middle part of the padded image and return it as the filtered image
    #so we would have to constanly conver tindices back and forth from padded image to original image
    #we will iterate over the indices where the value is Inf, i.e indices_infinity
    for i in range(len(indices_infinity[0])):
        #get the indices of the patch
        row_index = indices_infinity[0][i]
        col_index = indices_infinity[1][i]
        #get the index on padded image
        row_index_padded = row_index + kernal_radius
        col_index_padded = col_index + kernal_radius
        #get the patch
        patch = padded_image[row_index_padded - kernal_radius:row_index_padded + kernal_radius + 1, \
            col_index_padded - kernal_radius:col_index_padded + kernal_radius + 1]
        #get the median of the patch
        #sort the patch
        patch = np.sort(patch, axis=None)
        #get the indexes where array has Inf
        indices_inf = np.where(np.isinf(patch))
        #if there is no value other than INf, we will continue
        if len(indices_inf[0]) == len(patch):
            continue
        #if there is no INf, then the index of array_end will be the last index
        elif len(indices_inf[0]) == 0:
            array_end = len(patch) - 1
        else:
            #get the index of first Inf
            array_end = np.where(np.isinf(patch))[0][0] - 1 
            #now array_end has to be >= 0 as we have already checked for the case where there is all Inf
            #and thus, the index of arrsy_end be non-negative
        #ge the array shortened to the index of array_end
        patch = patch[0:array_end + 1]
        #get the median
        median = np.median(patch)
        #replace Inf at teh index in padded image with median
        padded_image[row_index_padded, col_index_padded] = median
    #get the middle part of the padded image
    filtered_image = padded_image[kernal_radius:-kernal_radius, kernal_radius:-kernal_radius]
    return filtered_image

def initialize_median_filter(observed_image, application_matrix, kernal_radius= 1):
    _, c, h, w = observed_image.shape
    device = observed_image.device
    out = np.zeros((h,w,c))
    for i in range(c):
        initial_image = observed_image.permute(0, 2, 3, 1).contiguous().squeeze().clone().detach().cpu().numpy()[:, :, i]
        #we get the indices from the inpainting mask where the value is 0
        indices_mask = np.where(application_matrix.squeeze().clone().detach().cpu().numpy() == 0)
        #at all the indices  from indices_mask we will replace the value in observed_image with Inf
        initial_image[indices_mask] = np.inf
        #we will have a loop in which the kernel radius will be increased by 2
        while kernal_radius <= min(initial_image.shape[0], initial_image.shape[1])+2:
            #we will apply the median filter to the initial_image
            initial_image = Inf_median_filter_2D(initial_image, kernal_radius)
            #we will check if there are any Inf in the initial_image
            if np.isinf(initial_image).any():
                #if there are Inf, we will increase the kernal radius by 2
                kernal_radius = kernal_radius + 2
            else:
                #if there are no Inf, we will break the loop
                break
        out[:, :, i] = initial_image
    #we will return the initial_image
    return torch.from_numpy(out).float().to(device).permute(2,0,1).unsqueeze(0)


# IP SOLVER
def grad_f(x, b, op):
  # input: x- wavelet coeff.
  #        L- Lipchitz const.
  # out: grad_f(.) = At.A(.) - At(b)
  inr = op.A(x)-b
  grad = op.At(inr)
  return grad

class IP_solver:
    def __init__(self, params):
        """
        Initializes the PnP solver with the given regularization parameters and other parameters.
        Parameters:
        params (config): utils.config variable with instances
            - b: Distroted image
            - ground truth: ground truth for progress evaluation
            - L: Lipchitz constant
            - linear_op: Linear operator model
            - regularizer: Regularization model
            - max_iter: Maximum number of iterations
            - tol: Tolerance for stopping criterion
        """
        self.params = params
    
    def run_solver(self):
        return self.ISTA()
    
    def ISTA(self):
        params = self.params
        
        # Initialization
        xk = params.x0
        L = params.L
        sigma = params.sigma
        
        # Initialize forward operator
        lin_op = params.linear_op
        
        # Initialize denoiser
        prox_op = params.regularizer
        
        # Lists to store norms
        xk_norms = []
        error_norms = []
        
        for i in range(params.max_iter):
            # print(i)
            grad = grad_f(xk, params.b, lin_op)
            g = xk - (1/L)*grad
            
            if params.guide_iter == 0:
                xkp1 = prox_op(g, params.ground_truth, sigma)
            elif i < params.guide_iter:
                guide = g
                xkp1 = prox_op(g, guide, sigma)
            else:
                xkp1 = prox_op(g, guide, sigma)
            
            # Calculate and store norms
            xk_norm = np.linalg.norm(xkp1)
            error_norm = np.linalg.norm(xkp1 - params.ground_truth)
            
            xk_norms.append(xk_norm)
            error_norms.append(error_norm)
            
            xk = xkp1
        
        return xk, xk_norms, error_norms
    
    def FISTA(self):
        """
            params - b, mode, oper, L, max_iters \\
            reg_params - parameters for the regularizer 
        """

        params = self.params

        # Initialization
        xk_minus_1 = torch.zeros_like(params.b)
        xk = yk = torch.zeros_like(params.b)
        tk = 1
        L = params.L
        sigma = params.sigma

        # Initialize forward operator
        lin_op = params.linear_op

        # Initialize denoiser
        prox_op = params.regularizer

        func_vals = []
        for i in range(params.max_iter):
            xk = prox_op(yk - (1/L) * grad_f(yk, params.b, lin_op), sigma)
            tk_plus_1 = (1 + math.sqrt(1 + 4 * (tk**2)))/2
            yk = xk + ((tk - 1) / tk_plus_1) * (xk - xk_minus_1)

            xk_minus_1 = xk
            tk = tk_plus_1

        return yk, func_vals
