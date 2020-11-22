# Datasets

In this file, we provide a short description of datasets we plan to use and introduce the processing steps we have done on them so far.
Our goal is object detection in images captured by Unmanned Aerial Vehicles ('drones'). We plan to use two distinct datasets.

## The VisDrone DET-2019 dataset <sup>1,2 </sup>

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


<sup> 1 </sup> Vision meets drones: A challenge,
Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua,
arXiv preprint arXiv:1804.07437, 2018

<sup> 2 </sup> Vision Meets Drones: Past, Present and Future,
Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Hu, Qinghua and Ling, Haibin,
arXiv preprint arXiv:2001.06303, 2020

## MultiDrone Public DataSet <sup> 3 </sup>

The public MultiDrone Dataset has been assembled using both pre-existing audiovisual material and newly filmed UAV shots. A large subset of these data has been annotated for facilitating scientific research, in tasks such as visual detection and tracking of bicycles, football players, human crowds, etc.

A dataset for bicycle detection/tracking was assembled, consisting of 7 Youtube videos (resolution: 1920 x 1080) at 25 frames per second. Annotations are not exhaustive, i.e., there may be unannotated objects in the given video frames. The annotations are stored in the following format:



| channel | frameN | objectID | X1 | X2 | Y1 | Y2 | ObjectType/view |
|---------|--------|----------|----|----|----|----|-----------------|


The license agreement allows to use only for scientific research, testing and development purposes and we aren’t authorized to put any part of the dataset on the publicly accessible Internet. 

<sup> 3 </sup>  I. Mademlis, V. Mygdalis, N.Nikolaidis, M. Montagnuolo, F. Negro, A. Messina and I.Pitas, “High-Level Multiple-UAV Cinematography Tools for Covering Outdoor Events”, IEEE Transactions on Broadcasting, vol. 65, no. 3, pp. 627-635, 2019.
I. Mademlis, N.Nikolaidis, A.Tefas, I.Pitas, T. Wagner and A. Messina, “Autonomous UAV Cinematography: A Tutorial and a Formalized Shot-Type Taxonomy”, ACM Computing Surveys, vol. 52, issue 5, pp. 105:1-105:33, 2019.
