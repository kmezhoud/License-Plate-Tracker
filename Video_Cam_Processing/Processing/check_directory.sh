#!/bin/bash

path=$1

#folders=$(ls -d $path/*/ | xargs -n 1 basename)

folders=$(ls $path)
## looping in a list of directories that have a list of video
for d in $folders ; do
    # Check if the target is not a directory
    if [ ! -d "$d" ]; then
      echo " $d is not a directory"
      #exit 1
    else
    echo "$d is a directory" 
    fi
done
