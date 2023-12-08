#!/bin/bash

path=$1 
ext=$2 #mp4

#datetime=$(ls $path | grep -E $ext | awk -F '[_.]' '{print $4}' | grep -Eo ".{14}" | sort)

## Used with filename without prefix
datetime=$(ls $path | grep -E "mp4" | awk -F '[_.]' '{print $3}' | sort)

#datetime=$(ls $path)


counter=0
for dt in $datetime
do  
    f=$(ls $path | grep -E $dt)
    counter=$((counter + 1))
    #echo "$path/${counter}_${f}"
    
    ## remove prefix
    #mv "$path/$f" "$path/${f#*_}"
    ## add prefix
    mv "$path/$f" "$path/${counter}_${f}"
    ## add/replace prefix by new counter
    #mv "$path/$f" "$path/$counter_${f#*-}"
    # rename file replace - by _
    #mv -v "$path/$f" "$path/${f/.*{_}/}"
done
echo $counter
