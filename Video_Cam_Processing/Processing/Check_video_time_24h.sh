#!/bin/bash

# This script extract and sort/arrange datetime from videos filename.
# Loop throught video and extract duration.
# Sum duration and write informations to text file.


path=$1

folders=$(ls -d $path/*/ | xargs -n 1 basename)

for fld in $folders 
do 

    datetime=$(ls $path/$fld | grep -E "mp4" | awk -F '[_.]' '{print $4}' | grep -Eo ".{14}" | sort)

    #files=$(ls $path | grep -E "mp4")
    t=0
    times=(0)
    file_counter=0
    for dt in $datetime
    do
        # select file by order
        f=$(ls $path/$fld | grep -E $dt)
        #echo $f | grep -E "20231116"
        #ffmpeg -hide_banner -i "$path/$f"  2>&1 | grep Duration | cut -d ',' -f1
        #period=$(ffmpeg -hide_banner -i "$path/$fld/$f"  2>&1 | grep Duration | awk -F '[ ,]' '{print $4}') | sed 's/\.[^.]*$//' 
        #period=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -sexagesimal "$path/$fld/$f")

        t=$(ffmpeg -i "$path/$fld/$f" 2>&1 | grep "Duration" | grep -o " [0-9:.]*, " | head -n1 | tr ',' ' ' | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
        
        times+=("$t")
  
        #echo $_t | sed 's/ /+/g' | bc | awk '{printf("%d:%02d:%02d:%02d\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}'
        
        #echo "${times[@]}" | sed 's/ /+/g' | bc | awk '{printf("%d:%02d:%02d:%02d\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}'
        
        #dur=$(exiftool -n -q -p '${Duration;our $sum;$_=ConvertDuration($sum+=$_)
        #            }' "$path/$fld/$f" | tail -n1)
                    
        #duration+=("$dur")
        
        # save to file
        video_duration=$(echo $t | sed 's/ /+/g' | bc | awk '{printf("%d:%02d:%02d:%02d\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}')
        
        echo "Video Duration of:${dt:0:4}/${dt:4:2}/${dt:6:2},  started at ${dt:8:2}:${dt:10:2}:${dt:12:2} is : $video_duration and the sum = $(echo "${times[@]}" | sed 's/ /+/g' | bc | awk '{printf("%d:%02d:%02d:%02d\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}')" >> "$path/$(basename $path)_total_hours.txt"
    
        file_counter=$((file_counter+1))
    done 
    
    #echo "${times[@]}" | sed 's/ /+/g' | bc | awk '{printf("%d:%02d:%02d:%02d\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}'
    
    #echo $(date -d@"${period[@]}" | sed 's/ /+/g' | bc -u +%H:%M:%S)
    
    day_duration=$(echo "${times[@]}" | sed 's/ /+/g' | bc | awk '{printf("%dj:%02dh:%02dm:%02ds\n",($1/60/60/24),($1/60/60%24),($1/60%60),($1%60))}')
    
    # Print total duration of all video
    #echo "Video for day  duration of: $(basename $path), date: '$(basename ${fld%.*}), Number of Video: $file_counter is : $day_duration"
    # save to file
    #echo "Total video duration of day: $(basename $path), date: $(basename ${fld%.*}), Number of Video: $file_counter is : $day_duration" >> "$path/total_hour.txt"
    
    echo "====================================================================================================
            Cam: $(basename $path),  date: $(basename ${fld%.*}), Number of Videos: $file_counter,  Duration : $day_duration
    ====================================================================================================" >> "$path/$(basename $path)_total_hours.txt"
done
