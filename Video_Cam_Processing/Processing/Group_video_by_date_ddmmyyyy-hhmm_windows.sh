#!/bin/bash

## How to run
# cmd promp
#Set-Location -Path D:\Camera_without_duplicate\
#bash chmod +x .\Group_video_by_date_ddmmyyyy-hhmm.sh
# bash ./group_video_by_date.sh "./Cam6"

#path=/media/kirus/USB_DISK/Cam_surveillance/Cam16
#path=video/Cam16
path=$1

#unique_date=$(ls $path  | awk -F '[_.]' '{print $4}' | grep -Eo ".{8}" | sort -u)
unique_date=$(ls $path | awk -F '[-.]' '{print $1}' | sort -u)

for d in $unique_date
do
  SECONDS=0
  # make directory if not exists
  mkdir -p $path/../"$(basename ${path%.*})_sorted"
  mkdir -p $path/../"$(basename ${path%.*})_sorted"/$d
  # Filter file by specific date
  files=$(ls $path  | grep -E "$d")
  echo "creating folder $d, and copying files"
  for f in $files
  do
  #move file to new directory
  cp -r "$path/$f" "$path/../$(basename ${path%.*})_sorted/$d"
  done
done