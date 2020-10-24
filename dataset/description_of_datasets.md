# Datasets

In this file, we provide a short description of datasets we plan to use and introduce the processing steps we have done on them so far.
Our goal is object detection in images captured by Unmanned Aerial Vehicles ('drones'). We plan to use two distinct datasets.

## The VisDrone DET-2019 dataset <sup>1,2

This dataset consits of images taken by different UAVs under various conditions, eg. in urban and rural areas,
under varying lightning and weather conditions, from different heigths, with few or many objects in one picture.

For the images, we have annotations that contain
+ bounding box coordinates
+ the class of the object (eg. car, pedestrian, etc.)
+ additional information (eg. parts of the object outside the frame)

The dataset comes with a pre-defined train/validation/test split consisting of 6471, 548, 1610 images respectively. The dataset is freely available under
https://github.com/VisDrone/VisDrone-Dataset

The dataset was collected by the AISKYEYE team at Lab of Machine Learning and Data Mining , Tianjin University, China. 
Although the linked repo is unclear about licensing, we believe that the data is free to be used for research purposes.


<sup>1<\sup> Vision meets drones: A challenge,
Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua,
arXiv preprint arXiv:1804.07437, 2018

<sup>2<\sup> Vision Meets Drones: Past, Present and Future,
Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Hu, Qinghua and Ling, Haibin,
arXiv preprint arXiv:2001.06303, 2020
