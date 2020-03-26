# CSCE636-project-porta
## Overview

This repo contains all the code for the CSCE636 project.

I have taken a transfer learning approach to the problem, using a 3D Resnet trained on the Kinetics data set. 

Since the weights have already been highly refined, only a small data set from my domain is required (leaving the house) my training set is [here](https://youtu.be/p8XmS7LYZ8Q). My validation set is [here](https://youtu.be/FYVzdUH5qqc), and a training set is [here](https://youtu.be/aaDWt2fIZXo). As the semester progresses, I will diversify the data set further. The relevant videos need not be processed, they have already been converted to a string of .jpg files, which are contained in this repo. 

![Closing the door (returning)](https://raw.githubusercontent.com/tdgriffith/CSCE636-project-porta/master/videos/jpg_door/train/returning/p8XmS7LYZ8Q_000108_000112/image_00011.jpg)

## Instructions for use
I highly recommend a virtual environment, you never know.
- Required packages:
    - Clone this repo
    - Use the included environment-636.yml file to downloaded required packages
        - It's a little heavy, but I will thin it out over the rest of the semester
    - Download the pre-trained weights from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M)
        - Specifically, the weights used in the notebook are in resnext-101-kineteics.pth
        - Place this .pth file in the root directory
- Pre-trained notes
    - Heavily borrowed from [Kensho Hara's work](https://kenshohara.github.io/)
    - [Original repo here](https://github.com/kenshohara/3D-ResNets-PyTorch)
- Data set notes
    - I am using the [ActivityNet Data Crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) to pull my own data set from Youtube in a network friendly format. 
    - This crawler pulls videos from Youtube, and converts them to a series of jpg images, which are fed to the network. Images are already processed and contained in this repo
- Final model state_dict
    - Download [here](https://drive.google.com/drive/folders/1euUdVgU2mKkHPt5yPUWRuAMBLhelr_Ba?usp=sharing) from Drive.

