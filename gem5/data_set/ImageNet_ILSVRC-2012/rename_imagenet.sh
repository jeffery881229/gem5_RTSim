#!/bin/bash
cd ~/gem5/data_set/ImageNet_ILSVRC-2012/imagenet_bin_outputs

for file in imagenet_img*.bin; do
    base=$(basename "$file" .bin)
    id=$(echo $base | cut -d'_' -f3 | sed 's/^0*//')
    res=$(echo $base | cut -d'_' -f4)
    newname="imagenet_img${id-1}_${res}.bin"
    mv "$file" "$newname"
    echo "Renamed $file -> $newname"
done

