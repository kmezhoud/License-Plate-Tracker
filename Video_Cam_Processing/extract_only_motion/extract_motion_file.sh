#!/bin/bash
## How to run
## ./extract_motion_file.sh path
#path="/media/DATA/ComputerVision/video/Cam6"
path=../video/Cam6/all_video.mp4

SECONDS=0
dvr-scan -i $path \
         -o $(basename ${path%.*})_motion.avi \
         -bb \
         -d ../video/Cam6/motion \
         -t 0.15 \
         --logfile "../video/Cam6/log/log_$(basename ${path%.*}).txt"
duration=$SECONDS
echo "ELAPSED TIME FOR $(basename ${path%.*}): $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed." >> "../video/Cam6/log/log_$(basename ${path%.*}).txt"

