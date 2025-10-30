import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from glob import glob
from torch.utils.data import DataLoader, Dataset

def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros (arr.shape + ( len ( channels ),), dtype = dtype )
    for c in channels:
        c = int (c)
        res[..., c:c+1][arr == c] = 1

    return res


def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                  getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading prematurely, leaves arrays mostly empty for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    img = nib.load(imageNames[0])
    first_case = img.get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case [:, :, 0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype )
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm ( inImage )
            #~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else :
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images
    

# Modified from https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing#scrollTo=d9842512
# and from the response given by chatGPT in prompt 2 of aiUsage.txt
class DataSegmenter2D:
    """TODO."""

    def __init__(self, image_path, mask_path, subset_size=None, start_index=0):
        """
        Initialize the dataset, or a subset of the dataset, from image and mask path patterns.
        
        Args:
            image_path (str): Glob path to image files, e.g. 'Data/.../train/*.nii.gz'
            mask_path (str): Glob path to mask files, e.g. 'Data/.../seg_train/*.nii.gz'
            subset_size (int or None): If set, limits how many samples are loaded.
            start_index (int): The starting index of the images to load. Images before this index are not loaded.
        """

        self.image_paths = sorted(glob(image_path))
        self.mask_paths = sorted(glob(mask_path))

        if (len(self.image_paths) != len(self.mask_paths)):
            print(f"Warning. The number of images ({len(self.image_paths)}) does not equal the number of masks ({len(self.mask_paths)})")

        self.dataset_size = min(len(self.image_paths), len(self.mask_paths))

        # Remove elements before the starting index
        if (self.dataset_size <= start_index):
            print(f"Warning: start index ({start_index}) >= the size of the data ({size}). No data will be stored.")
        self.image_paths = self.image_paths[start_index:]
        self.mask_paths = self.mask_paths[start_index:]

        # Update the size after removing elements
        self.dataset_size = min(len(self.image_paths), len(self.mask_paths))
        print(f"There are {self.dataset_size} samples beyond the start index")

        # Use subset if specified
        if subset_size is not None and subset_size < self.dataset_size:
            # If we're using a subset, we reduce the number of images.
            
            self.image_paths = self.image_paths[:subset_size]
            self.mask_paths = self.mask_paths[:subset_size]
            
            # Print first to use the old size.
            print(f"Using subset of {subset_size} samples (out of {self.dataset_size} total)")
            self.dataset_size = min(len(self.image_paths), len(self.mask_paths))
        else:
            print(f"Using all {self.dataset_size} samples")
        
        # Loads the subset into memory
        self.images = load_data_2D(self.image_paths, normImage=True)
        self.masks = load_data_2D(self.mask_paths, dtype=np.uint8)

        print("Images shape: ", self.images.shape)
        print("Masks shape: ", self.masks.shape)
        

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Get image and mask
        return self.images[idx], self.masks[idx]


# Get file lists
# These image paths require being called in UNetMatthewMoore
train_img_path = "Data/HipMRI_Study_open/keras_slices_data/keras_slices_train/*.nii.gz"
train_mask_path = "Data/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/*.nii.gz"

val_img_path = "Data/HipMRI_Study_open/keras_slices_data/keras_slices_validate/*.nii.gz"
val_mask_path = "Data/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/*.nii.gz"

print("Creating training data segmenter")
dataSegmenter = DataSegmenter2D(train_img_path, train_mask_path, subset_size=800)

print("Creating evaluator data segmenter")
dataSegmenter = DataSegmenter2D(val_img_path, val_mask_path, subset_size=400)