# Segmenting Hip MRI study data.

This project attempts to segment Hip MRI study data to identify different segments of the data.
The data has been segmented by a MR physicist into 6 categories; Empty, Body outline, Bone, Bladder, Rectum, and Prostate.


# Algorithm:
This project segments the data using the UNet structure. The UNet architecture is a modified version of
the autoencoded architecture, featuring skip connections between layers.
The UNet implementation in this project uses three encoding convolution layers, and three decoding convolution layers.
The first encoding layer has a skip connection to the last decoding layer, the second 
encoding layer has a skip connection to the second last decoding layer, and so on.

Each convolution layer contains several steps, namely:
Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout


# Results:
The results of the project were disappointing. After 100 epochs of training, the Dice loss for each segment type was:

Dice Scores per Class:
  Empty       : 0.9639
  Body Outline: 0.9495
  Bone        : 0.7777
  Bladder     : 0.9074
  Rectum      : 0.5926
  Prostate    : 0.6389
Mean Dice Score: 0.8050

The average dice loss over time was as follows.

![Description of image]("images/dice_loss_chart.png)

While these results seem to be slowly improving, looking at the data shows that this is misleading.
For example, these are images representing the segmentation around epoch 60 (Before the large drop in dice loss)

![Before drop]("images/epoch_60_data_1.png")

![Before drop]("images/epoch_60_data_2.png")

As seen, while somewhat inaccurate, it generally identifies each segment. In contrast, here is an image from epoch
100 (the final model)


![Before drop]("images/epoch_100_data_1.png")

![Before drop]("images/epoch_100_data_2.png")

As seen, the smaller segments (the prostate and rectum) are simply ignored. Due to their rarity and small size,
the loss function likely punishes the model more for wrongly identifying these two organs than for identifying them.
As such, by simply ignoring them, their average scores increase. This is likely the cause of the sharp drop in loss.

# Data requirements:
The Hip MRI data was downloaded from the uq Rangpur, at the file path:
/home/groups/comp3710/HipMRI_Study_open
Alternatively, the files can also be found here: https://data.csiro.au/collection/csiro:51392v2?redirected=true
The file paths returned from the get_image_path_hip_mri function in dataset.py 
expect the data to be in the structure.

Data
    HipMRI_Study_open
        keras_slices_data
            keras_slices_seg_test
            keras_slices_seg_train
            keras_slices_seg_validate
            keras_slices_test
            keras_slices_train
            keras_slices_validate
        semantic_labels_only
        semantic_MRs

Where each folder contains the corresponding *.nii.gz data files, and
Data is in the folder that the python files are called from.
Only the files found inside keras_slices_data are required. If a different file
structure is used, the paths in get_image_path_hip_mri in dataset.py will need
to be changed.

The data from rangpur came pre-segmented into testing, validation, and testing datasets.
Specifically, there are 11460 training samples, 660 validation samples, and 540 testing samples.
Since this distribution gave a reasonable distribution of data, it was kept.

# Requirements
torch torchvision numpy matplotlib tqdm nibabel

# References:

This project was built off of the 2D binary image classification example on blackboard.
A link to this example's source code can be found here:
https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1


Various pytorch documentation sites were referenced. Most notably were the following:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

ChatGPT was also heavily used throughout the project. A full transcript of all prompts
and responses can be found in aiUsage.txt