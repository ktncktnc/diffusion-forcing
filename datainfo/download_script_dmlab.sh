#!/bin/bash

for part in aa ab ac
do
    echo "Downloading ${part}"
    aria2c -s 16 -x 16 -d /content/drive/MyDrive/datasets/dmlab "https://archive.org/download/dmlab_dataset_${part}/dmlab.tar.part${part}"
done