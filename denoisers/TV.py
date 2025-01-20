import torch

class TV:
    def __init__(self, lamda, iters, l, u, proj_p):
        self.ops = L_image_grad()
        self.lamda = lamda
        self.iters = iters
        self.l = l
        self.u = u
        self.proj_p = proj_p

    def __call__(self, x, guide=None, sigma=None):
        """
        FGP implementation for color images with channels processed in parallel using PyTorch.

        Args:
            x: Input color image tensor of shape (C, M, N)

        Returns:
            Processed color image tensor of shape (C, M, N)
        """
        # Check if input has valid shape
        outdim=3
        # print(x.shape)
        if len(x.shape) != 3:
            outdim=4
            x = x.squeeze()
            assert len(x.shape) == 3, "Input image must have shape (C, M, N)"

        c, m, n = x.shape
        device = x.device
        y_hat = torch.zeros_like(x)

        # Initialize variables for all channels in parallel
        pk_minus_1 = rk = torch.zeros((c, m - 1, n), dtype=torch.float32, device=device)
        qk_minus_1 = sk = torch.zeros((c, m, n - 1), dtype=torch.float32, device=device)
        tk = torch.ones((c,), dtype=torch.float32, device=device)

        for i in range(self.iters):
            # Apply FGP algorithm for all channels in parallel
            Pc = TV_proj_c(x - self.lamda * self.ops.L((rk, sk)), self.l, self.u)
            temp_p, temp_q = self.ops.L_T(Pc)
            temp_p = rk + (1 / (8 * self.lamda)) * temp_p
            temp_q = sk + (1 / (8 * self.lamda)) * temp_q
            pk, qk = TV_proj_p((temp_p, temp_q), self.proj_p)

            tk_plus_1 = (1 + torch.sqrt(1 + 4 * tk**2)) / 2

            rk = pk + (tk.view(-1, 1, 1) - 1) / tk_plus_1.view(-1, 1, 1) * (pk - pk_minus_1)
            sk = qk + (tk.view(-1, 1, 1) - 1) / tk_plus_1.view(-1, 1, 1) * (qk - qk_minus_1)

            pk_minus_1 = pk
            qk_minus_1 = qk
            tk = tk_plus_1

        # Calculate the final result for all channels in parallel
        y_hat = TV_proj_c(x - self.lamda * self.ops.L((pk, qk)), self.l, self.u)
        if outdim==4:
            return y_hat.unsqueeze(0)
        return y_hat

def TV_proj_p(p_q, proj_p):
    p = p_q[0]
    q = p_q[1]

    if proj_p == "isotropic":
        c, m, n = q.shape
        mag_grad = torch.sqrt(p[:, :m-1, :n-1]**2 + q[:, :m-1, :n-1]**2)
        p[:, :m-1, :n-1] = p[:, :m-1, :n-1] / torch.maximum(torch.tensor(1.0, dtype=torch.float32), mag_grad)
        p[:, :, n-1] = p[:, :, n-1] / torch.maximum(torch.tensor(1.0, dtype=torch.float32), torch.abs(p[:, :, n-1]))
        q[:, :m-1, :n-1] = q[:, :m-1, :n-1] / torch.maximum(torch.tensor(1.0, dtype=torch.float32), mag_grad)
        q[:, m-1, :] = q[:, m-1, :] / torch.maximum(torch.tensor(1.0, dtype=torch.float32), torch.abs(q[:, m-1, :]))

    elif proj_p == "l1-term":
        p = p / torch.maximum(torch.tensor(1.0, dtype=torch.float32), torch.abs(p))
        q = q / torch.maximum(torch.tensor(1.0, dtype=torch.float32), torch.abs(q))

    return (p, q)

def TV_proj_c(image, l, u):
    """
    Project tensor values to range [l, u]
    
    Args:
        image: Input tensor
        l: Lower bound
        u: Upper bound
    Returns:
        Projected tensor with values clamped between l and u
    """
    return torch.clamp(image, min=l, max=u)

class L_image_grad:
    def __init__(self):
        pass
        
    def L_T(self, image):
        """
        Transpose gradient operator
        Args:
            image: Input tensor of shape (c, m, n)
        Returns:
            tuple of tensors (p, q) containing vertical and horizontal differences
        """
        c, m, n = image.shape
        p = image[:, :m-1, :] - image[:, 1:m, :]
        q = image[:, :, :n-1] - image[:, :, 1:n]
        return (p, q)

    def L(self, p_q):
        """
        Gradient operator
        Args:
            p_q: tuple of tensors (p, q) containing vertical and horizontal components
        Returns:
            tensor of shape (c, m, n)
        """
        p, q = p_q
        c, m, n = q.shape[0], p.shape[1] + 1, q.shape[2] + 1
        x = torch.zeros((c, m, n), dtype=p.dtype, device=p.device)
        
        x[:, :m-1, :] += p
        x[:, 1:m, :] -= p
        x[:, :, :n-1] += q
        x[:, :, 1:n] -= q
        return x
