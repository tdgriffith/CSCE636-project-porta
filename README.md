# CSCE636-project-porta
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

