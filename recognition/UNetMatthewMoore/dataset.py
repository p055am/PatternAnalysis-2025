import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from glob import glob

def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros (arr.shape + ( len ( channels ),), dtype = dtype )
    for c in channels:
        c = int (c)
        res[..., c:c+1][arr == c] = 1

    return res

'''
Load medical image data from names, cases list provided into a list for each.

This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

normImage: bool (normalise the image 0.0-1.0)
early_stop: Stop loading prematurely, leaves arrays mostly empty for quick loading and testing scripts.
'''
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                  getAffines=False, early_stop=False):
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


# Get file lists
# These image paths require being called in UNetMatthewMoore
train_img_files = sorted(glob("Data/HipMRI_Study_open/keras_slices_data/keras_slices_train/*.nii.gz"))
train_mask_files = sorted(glob("Data/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/*.nii.gz"))

val_img_files = sorted(glob("Data/HipMRI_Study_open/keras_slices_data/keras_slices_validate/*.nii.gz"))
val_mask_files = sorted(glob("Data/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/*.nii.gz"))


#Gets the number of files in the list.
print("Number of training image files: ", len(train_img_files))
print("Number of training mask files: ", len(train_mask_files))

print("Number of evaluation image files: ", len(val_img_files))
print("Number of evaluation mask files: ", len(val_mask_files))

# Load into memory (can use early_stop=True while testing)
x_train = load_data_2D(train_img_files, normImage=True, early_stop=True)
y_train = load_data_2D(train_mask_files, dtype=np.uint8, early_stop=True)

x_val = load_data_2D(val_img_files, normImage=True, early_stop=True)
y_val = load_data_2D(val_mask_files, dtype=np.uint8, early_stop=True)

print("X train shape: ", x_train.shape)
print("Y train shape: ", y_train.shape)

print("X val shape: ", x_val.shape)
print("Y val shape: ", y_val.shape)