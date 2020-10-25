# Object detection in drone-images
### Homework Project of team 'zöld137'

Our project aims to detect various objects in drone-captured images.

We would like to solve this problem as a supervised task, so acquired two datasets containing pictures taken by drones. ('VisDrone' and 'MultiDrone'.)
Both sets come with annotations, what means coordinates for bounding boxes and the ground-truth classes of annotated objects.

Few words about the repository structure below.

## zold137/datasets

This directory contains our code and documentation for data acquisition and basic processing.

__description_of_datasets__ is our writeup about the data with a self-explaining name.

In addition to this, we provide basic visualization in Jupyter Notebooks 'Saving_frames_and_drawing_boxes.ipynb' and 'Data_inspection.ipynb'
The main purpose of these notebooks is to introduce the datasets and give some impression about them. As both sets consist of numerous pictures, they are not uploadaded to GitHub. 

Thus, one may not want to run the code, only look at the visualized examples.

## zold137/conv_tests

This directory is __not__ part of the 1<sup>st</sup> milestone. We started to experiment with renowned CNNs if they correctly classify objects cropped from the ground-truth bounding boxes. 

May contain sharp edges, it is not recommended to look at it now.


