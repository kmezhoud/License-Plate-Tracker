#!/bin/bash

## How to run
##  ps> bash ./Concatenate_video.sh ./Cam16_sorted/20231108/motion Cam16

path=$1
camera=$2

#unique_date=$(ls $path  | awk -F '[_.]' '{print $4}' | grep -Eo ".{8}" | sort -u)

datetime=$(ls $path | awk -F '[_.]' '{print $4}' | grep -Eo ".{14}" | sort)

#echo $datetime
# Generate input file with a list of video file name to cancatenate
for dt in $datetime
do
    #echo $dt
    f=$(ls $path | grep -E $dt)
    #echo $(dirname $path | xargs basename)
    file_list_name=$(echo $path/$(dirname $path | xargs basename)_motion_list.txt)
    echo "file '$f'" >> $file_list_name
done
# or with printf
#printf "file '%s'\n" $path/*.$format > "$(basename $path)_list.txt"
date=$(echo $dt | grep -Eo ".{8}")
# Cancatenate the list of file in mylist.txt using ffmpeg
# https://ffmpeg.org/ffmpeg-formats.html#concat-1
ffmpeg.exe -f concat -safe 0 -i $file_list_name -c copy "$path/${camera}_${date}_motion_all.avi"
#ffmpeg -f concat -safe 0 -i "$(basename $path)_list.txt" -c copy ${PWD##*/}_all.mp4

## For Windows
#(echo file 'first file.mp4' & echo file 'second file.mp4' )>list.txt
#ffmpeg -safe 0 -f concat -i list.txt -c copy output.mp4


## Cut video
#ffmpeg -ss 00:00:00 -i cam16-20231114-all.avi -t 03:07:09 -codec copy cam16-20231114.avi
