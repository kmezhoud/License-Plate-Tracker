#!/bin/bash
# For Windows
## How to run
## <PowerShell D:\camera> bash ./Extract_only_motion.sh "./Cam6_sorted/20231108"
## option for cam16: -a 0 9 595 48 620 265 1280 384 1276 720 963 720 3 716 \
## option 2 without small door: -a 5 33 415 54 486 126 491 236 1280 337 1280 720 4 717

path=$1

#files=$(gci -File $path)
files=$(ls $path)

for f in $files
do
    #echo $f
    #echo "basename: $(basename ${f%.*})"
    #echo $path
    # check if file is done
   if test -f "$path/motion/$(basename ${f%.*})_motion.avi"
    then
      echo "File exists: $(basename ${f%.*})_motion.avi"
    else

    SECONDS=0
   #echo $($path${f%.*})
    dvr-scan.exe -i "$path/$f" \
           -o "$(basename ${f%.*})_motion.avi" \
           -bb \
           -d "$path/motion" \
           -t 0.35 \
           -a 5 33 415 54 486 126 491 236 1280 337 1280 720 4 717 \
           --logfile "$path/log/log_$(basename ${f%.*}).txt"
    duration=$SECONDS
    echo "ELAPSED TIME FOR $(basename ${f%.*}): $(($duration / 60)) minutes and $(($duration % 60)) seconds." >> "$path/log/log_$(basename ${f%.*}).txt"
  fi
done
