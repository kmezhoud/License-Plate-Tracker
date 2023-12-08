#!/bin/bash
## How to run
## ./extract_only_motion.sh path
#path="/media/DATA/ComputerVision/video/Cam6"
# #-a 0 9 595 48 620 265 1280 384 1276 720 963 720 3 716
path=$1 #" /media/DATA/ComputerVision/video/Cam6"
codec=$2 # mp4


for f in $path/*.$codec
do
    # check if file is done
    if test -f "$path/motion/$(basename ${f%.*})_motion.avi"
    then
      echo "File exists: $(basename ${f%.*})_motion.avi"
    else
    echo $f
    SECONDS=0
    dvr-scan -i $f \
           -o "$(basename ${f%.*})_motion.avi" \
           -bb \
           -d "$path/motion" \
           -t 0.35 \
           --logfile "$path/log/log_$(basename ${f%.*}).txt"
    duration=$SECONDS
    echo "ELAPSED TIME FOR $(basename ${f%.*}): $(($duration / 60)) minutes and $(($duration % 60)) seconds." >> "$path/log/log_$(basename ${f%.*}).txt"
  fi
done
