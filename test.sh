#!/bin/bash

GIT_REPO_URL="https://github.com/tdgriffith/CSCE636-project-porta.git"
REPO="CSCE636-project-porta"
VIDEO="test_video1.mp4"
VIDEO_name="test_video1"
UIN_JSON="627008000.json"
UIN_PNG="627008000.png"

#git clone $GIT_REPO_URL

#cd $REPO

echo $VIDEO
python utils/video_jpg_kinetics.py test test_proc
python utils/n_frames_kinetics.py test_proc
python test_pretrained.py preWeights/porta.pth test_proc/test/$VIDEO_name

cp test_results/test_results.json $UIN_JSON
cp test_results/test_results.png $UIN_PNG