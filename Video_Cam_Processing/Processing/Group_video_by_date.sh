#!/bin/bash

# extract unqiue date
# Loop throught video and move them to corresponding folder by date.

#path=/media/kirus/USB_DISK/Cam_surveillance/Cam16
#path=video/Cam16
path=$1

unique_date=$(ls $path  | awk -F '[_.]' '{print $4}' | grep -Eo ".{8}" | sort -u)

for d in $unique_date
do
  SECONDS=0
  # make directory if not exists
  mkdir -p $path/../"$(basename ${path%.*})_sorted"
  mkdir -p $path/../"$(basename ${path%.*})_sorted"/$d
  # Filter file by specific date
  files=$(ls $path  | grep -E "$d")
  echo "Creating folder $d, and copying files"
  for f in $files
  do
  #move file to new directory
  cp -r "$path/$f" "$path/../$(basename ${path%.*})_sorted/$d" 
  done
done
