#!/bin/bash

for part in aa ab ac ad ae af ag ah ai aj ak
do
    echo "Downloading ${part}"
    aria2c -s 16 -x 16 -d /content/drive/MyDrive/datasets/minecraft "https://archive.org/download/minecraft_marsh_dataset_${part}/minecraft.tar.part${part}"
done