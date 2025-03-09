from PIL import Image, ImageOps
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *

def augmentation(x, k=0, inverse=False):
    """Augment patch with rotation/flip"""
    k = k % 8
    if inverse:
        k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=(k//2) % 4, dims=[1, 2])

def load_images(path, color_mode='gray', subdirs=None, size=None):
    """
    Generic function to load images from directory/subdirectories
    Args:
        path (str): Root directory path
        color_mode (str): 'gray' or 'rgb'
        subdirs (list): Optional list of subdirectories to search in.
                       If None, searches in root path directly.
    Returns:
        list: List of loaded images as numpy arrays
    """
    images = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    # Helper function to process a single directory
    def process_directory(dir_path):
        local_images = []
        for root, _, files in os.walk(dir_path):
            for f in files:
                # Skip hidden files and invalid extensions
                if f.startswith('.'):
                    continue
                if os.path.splitext(f.lower())[1] not in valid_extensions:
                    continue

                try:
                    filepath = os.path.join(root, f)
                    if color_mode == 'gray':
                        img = (ImageOps.grayscale(Image.open(filepath)))
                    else:  # rgb
                        img = (Image.open(filepath).convert('RGB'))
                        # if len(img.shape) != 3 or img.shape[2] != 3:
                        #     print(f"Skipping {f}: Invalid RGB shape {img.shape}")
                        #     continue
                    if size:
                        img = np.array(img.resize((size, size)))
                    else:
                        img = np.array(img)
                    local_images.append(img.astype(np.uint8))
                except Exception as e:
                    print(f"Could not load {f}: {e}")
        return local_images
    
    if subdirs is not None:
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            if not os.path.exists(subdir_path):
                print(f"Warning: {subdir} not found in {path}")
                continue
            subdir_images = process_directory(subdir_path)
            images.extend(subdir_images)
            print(f"Loaded {len(subdir_images)} images from {subdir}")
    else:
        # Process root directory directly
        images = process_directory(path)

    if len(images) == 0:
        raise RuntimeError(f"No valid images found in {path}")

    if subdirs:
        print(f"Total loaded {len(images)} images from: {', '.join(subdirs)}")
    else:
        print(f"Total loaded {len(images)} images from: {path}")
    return images

class Train_img_Dataset(Dataset):
    def __init__(self, in_folder, color_mode='gray', sigma_min=15, sigma_max=15, patch_size=70, 
                 imgs_per_epoch=128, req_patches=True, factor=None, size=None,
                 no_patches=None, train_subdirs = ['Set12']):
        """
        Training dataset that matches demo_training.py implementation
        Args:
            in_folder (str): Input folder path
            color_mode (str): 'gray' or 'rgb'
            sigma_min (float): Min noise sigma (0-255)
            sigma_max (float): Max noise sigma (0-255)

            patch_size (int): Training patch size
            batch_size (int): Batch size (used for __len__)
            device: Device to put tensors on
        """
        super().__init__()

        if req_patches:
            if factor==None and no_patches==None:
                raise ValueError("input one out of 'factor' or 'no_of_patches' values")
            elif factor!=None and no_patches!=None:
                raise ValueError("input one out of 'factor' or 'no_of_patches' values")
            elif factor!=None:
                self.factor=factor
                self.no_patches= None
            else:
                self.no_patches= no_patches
                self.factor= None

        self.color_mode = color_mode
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.patch_size = patch_size
        self.imgs_per_epoch = imgs_per_epoch
        self.req_patches = req_patches
        # self.device = device

        # Training subdirectories to search

        # Load all images
        print("\nLoading training images...")
        self.images_train = load_images(in_folder, color_mode, train_subdirs, size=size)

        # Filter by size
        valid_images = []
        if type(patch_size) == int:
            for img in self.images_train:
                h, w = img.shape[:2]
                if h >= patch_size and w >= patch_size:
                    valid_images.append(img)
        else:
            for img in self.images_train:
                valid_images.append(img)
            
        self.images_train = valid_images
        self.number_of_images = len(self.images_train)

        if self.number_of_images == 0:
            raise RuntimeError(f"No valid images found in {in_folder}")
        print(f"Loaded {self.number_of_images} images for training")

    def __len__(self):
        return self.imgs_per_epoch  # As in demo_training.py

    def __getitem__(self, idx):
        img = self.images_train[idx]

        # Convert to float
        img = img.astype(np.float32) / 255.0

        # Random sigma
        sigma = np.random.uniform(
            low=self.sigma_min/255, high=self.sigma_max/255)

        # Get dimensions accounting for color mode
        if self.color_mode == 'gray':
            h, w = img.shape
        else:
            h, w, _ = img.shape

        # Add noise
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)

        # Convert to torch tensors
        if self.color_mode == 'gray':
            clean = torch.from_numpy(img).unsqueeze(0)
            noisy = torch.from_numpy(noisy_img).unsqueeze(0)
        else:
            clean = torch.from_numpy(img.transpose(2, 0, 1))
            noisy = torch.from_numpy(noisy_img.transpose(2, 0, 1))

        if self.req_patches:
            clean_patches = self._extract_random_patches(clean)
            b, c, h, w = clean_patches.shape
            clean = clean_patches[:,:,h//2,w//2]
            noisy = clean_patches + float(sigma)*torch.randn_like(clean_patches)
        # else:
        #     i = np.random.randint(0, h - self.patch_size - 1)
        #     j = np.random.randint(w - self.patch_size - 1)
        #     clean = clean[:, i:i+self.patch_size, j:j+self.patch_size]
        #     noisy = clean + float(sigma)*torch.randn_like(clean)   

        sigma_torch = sigma * torch.ones(1, 1, 1)

        return clean, noisy, sigma_torch

    def _extract_random_patches(self, img):

        patch_rad = self.patch_size//2
        C, H, W = img.shape

        # Total number of pixels and 30% selection
        total_pixels = H * W
        if self.factor != None:
            num_points = int(self.factor * total_pixels)
        else:
            num_points = self.no_patches

        # Randomly select points
        y_coords = torch.randint(patch_rad, H - patch_rad, (num_points,))
        x_coords = torch.randint(patch_rad, W - patch_rad, (num_points,))

        # Extract patches
        img_patches = list_of_pixels2patches(img, y_coords, x_coords, patch_rad)

        return torch.stack(img_patches)

class TrainingDataset(Dataset):
    def __init__(self, in_folder, color_mode='gray', sigma_min=15, sigma_max=15, patch_size=70, 
                 imgs_per_epoch=128, req_patches=True, no_patches = 128, size=None):
        """
        Training dataset that matches demo_training.py implementation
        Args:
            in_folder (str): Input folder path
            color_mode (str): 'gray' or 'rgb'
            sigma_min (float): Min noise sigma (0-255)
            sigma_max (float): Max noise sigma (0-255)
            patch_size (int): Training patch size
            imgs_per_epoch (int): Batch size (used for __len__)
            device: Device to put tensors on
        """
        super().__init__()

        self.factor = None
        self.no_patches = no_patches
        self.color_mode = color_mode
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.patch_size = patch_size
        self.imgs_per_epoch = imgs_per_epoch
        self.req_patches = req_patches
        # self.device = device

        # Training subdirectories to search
        train_subdirs = ['BSD400']
        train_subdirs = ['BSD400', 'DIV2K', 'Flickr2K', 'waterloo_exploration']

        # Load all images
        print("\nLoading training images...")
        self.images_train = load_images(in_folder, color_mode, train_subdirs, size=size)

        # Filter by size
        valid_images = []
        for img in self.images_train:
            h, w = img.shape[:2]
            if h >= patch_size and w >= patch_size:
                valid_images.append(img)

        self.images_train = valid_images
        self.number_of_images = len(self.images_train)

        if self.number_of_images == 0:
            raise RuntimeError(f"No valid images found in {in_folder}")
        print(f"Loaded {self.number_of_images} images for training")

    def __len__(self):
        return self.imgs_per_epoch  # As in demo_training.py

    # def __getitem__(self, idx):
    #     # Random sigma
    #     sigma = np.random.uniform(
    #         low=self.sigma_min/255, high=self.sigma_max/255)

    #     # Random image
    #     img = self.images_train[np.random.choice(self.number_of_images)]

    #     # Get dimensions accounting for color mode
    #     if self.color_mode == 'gray':
    #         h, w = img.shape
    #     else:
    #         h, w, _ = img.shape

    #     # Extract patch based on color mode
    #     img = img.astype(np.float32) / 255.0

    #     # Convert to torch tensors with proper shape
    #     if self.color_mode == 'gray':
    #         clean = torch.from_numpy(img).unsqueeze(0)
    #     else:
    #         clean = torch.from_numpy(img.transpose(2, 0, 1))

    #     # Random augmentation
    #     k = np.random.randint(8)
    #     clean = augmentation(clean, k)

    #     if self.req_patches:
    #         clean_patches = self._extract_random_patches(clean)
    #         b, c, h, w = clean_patches.shape
    #         clean = clean_patches[:,:,h//2,w//2]
    #         noisy = clean_patches + float(sigma)*torch.randn_like(clean_patches)
    #         # Create sigma tensor
    #         sigma_torch = sigma * torch.ones(b, 1, 1, 1)
    #     else:
    #         i = torch.randint(0, h - self.patch_size - 1)
    #         j = torch.randint(0, w - self.patch_size - 1)
    #         clean = clean[:, i:i+self.patch_size, j:j+self.patch_size]
    #         noisy = clean + float(sigma)*torch.randn_like(clean)   
    #         # Create sigma tensor
    #         sigma_torch = sigma * torch.ones(1, 1, 1)

    #     return clean, noisy, sigma_torch

    def __getitem__(self, idx):

        # Random image
        img = self.images_train[np.random.choice(self.number_of_images)]

        # Extract patch based on color mode
        img = img.astype(np.float32) / 255.0

        # Convert to torch tensors with proper shape
        if self.color_mode == 'gray':
            clean = torch.from_numpy(img).unsqueeze(0)
        else:
            clean = torch.from_numpy(img.transpose(2, 0, 1))

        # Random augmentation
        k = np.random.randint(8)
        clean = augmentation(clean, k)
        _, h, w = clean.shape

        if self.req_patches:
            clean_patches = self._extract_random_patches(clean)
            b, c, h, w = clean_patches.shape
            clean = clean_patches[:,:,h//2,w//2]
            # Random sigma
            sigma = self.sigma_min/255 + torch.rand(b,)*(self.sigma_max-self.sigma_min)/255
            noisy = clean_patches + (sigma[:,None, None, None]).expand_as(clean_patches) * torch.randn_like(clean_patches)
            sigma_torch = sigma[:,None].reshape(-1, 1, 1, 1)
        else:
            i = torch.randint(h - self.patch_size - 1, (1,)).item()
            j = torch.randint(w - self.patch_size - 1, (1,)).item()
            clean = clean[:, i:i+self.patch_size, j:j+self.patch_size]
            # Random sigma
            sigma = self.sigma_min/255 + torch.rand(1,)*(self.sigma_max-self.sigma_min)/255
            noisy = clean + (sigma[:,None, None]).expand_as(clean) * torch.randn_like(clean)
            sigma_torch = sigma * torch.ones(1, 1, 1)

        return clean, noisy, sigma_torch

    def _extract_random_patches(self, img):

        patch_rad = self.patch_size//2
        C, H, W = img.shape

        # Total number of pixels and 30% selection
        total_pixels = H * W
        if self.factor != None:
            num_points = int(self.factor * total_pixels)
        else:
            num_points = self.no_patches

        # Randomly select points
        y_coords = torch.randint(patch_rad, H - patch_rad, (num_points,))
        x_coords = torch.randint(patch_rad, W - patch_rad, (num_points,))

        # Extract patches
        img_patches = list_of_pixels2patches(img, y_coords, x_coords, patch_rad)

        return torch.stack(img_patches)

class ValidationDataset(Dataset):
    def __init__(self, val_dir, color_mode='gray', sigma=25):
        """
        Validation dataset that processes full images
        Args:
            val_dir (str): Validation directory path
            color_mode (str): 'gray' or 'rgb'
            sigma (float): Noise level (0-255)
        """
        super().__init__()

        self.color_mode = color_mode
        self.sigma = sigma / 255.0
        # self.device = device

        # Load all validation images
        self.images = load_images(val_dir, color_mode)
        if len(self.images) == 0:
            raise RuntimeError(f"No valid images found in {val_dir}")

        print(f"Loaded {len(self.images)} images for validation")

    def __len__(self):
        return len(self.images)  # Process all images

    def __getitem__(self, idx):
        img = self.images[idx]

        # Convert to float
        img = img.astype(np.float32) / 255.0

        # Add noise
        noise = np.random.normal(0, self.sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)

        # Convert to torch tensors
        if self.color_mode == 'gray':
            clean = torch.from_numpy(img).unsqueeze(0)
            noisy = torch.from_numpy(noisy_img).unsqueeze(0)
        else:
            clean = torch.from_numpy(img.transpose(2, 0, 1))
            noisy = torch.from_numpy(noisy_img.transpose(2, 0, 1))

        sigma_torch = self.sigma * torch.ones(1, 1, 1)

        return clean, noisy, sigma_torch

# Example usage:
def create_dataloaders(args):
    """Create training and validation dataloaders"""

    device = torch.device(
        f"cuda" if torch.cuda.is_available() else "cpu")

    # Training dataloader
    train_dataset = TrainingDataset(
        in_folder=os.path.join(args.in_folder, 'train'),
        color_mode=args.color_mode,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        # device=device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    # Validation dataloader (if validation directory exists)
    val_loader = None
    val_dir = os.path.join(args.in_folder, 'val/BSD32')

    if os.path.exists(val_dir):
        val_dataset = ValidationDataset(
            val_dir=val_dir,
            color_mode=args.color_mode,
            sigma=args.sigma_max/2,  # Middle of training range
            # device=device
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process one full image at a time
            shuffle=False,
            pin_memory=True
        )

    return train_loader, val_loader

class TestDataset(Dataset):
    def __init__(self, in_folder, color_mode='gray', sigma=15, show=False):
        """
        testing dataset that matches demo_testing.py implementation
        Args:
            in_folder (str): Input folder path
            color_mode (str): 'gray' or 'rgb'
            sigma_min (float): Min noise sigma (0-255)
            sigma_max (float): Max noise sigma (0-255)
            patch_size (int): testing patch size
            imgs_per_epoch (int): Batch size (used for __len__)
            device: Device to put tensors on
        """
        super().__init__()

        self.color_mode = color_mode
        self.sigma = sigma/255.0
        # self.imgs_per_epoch = imgs_per_epoch
        # self.device = device

        # testing subdirectories to search
        if show:
            test_subdirs = ['curr_test']
        else:
            test_subdirs = [d for d in os.listdir('./data/test_set') if os.path.isdir(os.path.join('./data/test', d))]
        # Load all images
        print("\nLoading testing images...")
        self.images_test = load_images(in_folder, color_mode, test_subdirs)

        self.number_of_images = len(self.images_test)

        if self.number_of_images == 0:
            raise RuntimeError(f"No valid images found in {in_folder}")
        print(f"Loaded {self.number_of_images} images for testing")

    def __len__(self):
        return len(self.images_test)  # Process all images

    def __getitem__(self, idx):
        img = self.images_test[idx]

        # Convert to float
        img = img.astype(np.float32) / 255.0

        # Add noise
        noise = np.random.normal(0, self.sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)

        # Convert to torch tensors
        if self.color_mode == 'gray':
            clean = torch.from_numpy(img).unsqueeze(0)
            noisy = torch.from_numpy(noisy_img).unsqueeze(0)
        else:
            clean = torch.from_numpy(img.transpose(2, 0, 1))
            noisy = torch.from_numpy(noisy_img.transpose(2, 0, 1))

        sigma_torch = self.sigma * torch.ones(1, 1, 1)

        return clean, noisy, sigma_torch
