#!/bin/bash

path=$1
codec=$2
#path=video/Cam16
#codec=mp4
# #-a 0 9 595 48 620 265 1280 384 1276 720 963 720 3 716
## Cam16 -a  5 33 415 54 486 126 491 236 128 337 1280 720 4 717 


folders=$(ls -d $path/*/ | xargs -n 1 basename)
#folders=$(ls $path)

## looping in a list of directories that have a list of video
for d in $folders ; do
    # Check if the target is not a directory
    if [ ! -d "$d" ]; then
    echo "$d seems to be not a folder. move it before to proceed"
      exit 1
    fi
    files=$(ls $d)
    
    for f in $files; do
    
      # check if file is done
      if test -f "$d/motion/$(basename ${f%.*})_motion.avi"
      then
        echo "File exists: $(basename ${f%.*})_motion.avi"
      else
     echo " f: $f"
     echo "base f: $(basename ${f%.*})"
     echo "path : $path"
     echo "d: $d"
      SECONDS=0
      #dvr-scan -i $f \
      #      -o "$(basename ${f%.*})_motion.avi" \
      #      -bb \
      #      -d "$d/motion" \
      #      -t 0.35 \
      #      --logfile "$d/log/log_$(basename ${f%.*}).txt"
      duration=$SECONDS
      echo "ELAPSED TIME FOR $(basename ${f%.*}): $(($duration / 60)) minutes and $(($duration % 60)) seconds." >> "$path/$d/log/log_$(basename ${f%.*}).txt"
    fi
  done
done
