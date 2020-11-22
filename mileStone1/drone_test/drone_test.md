# Drone test code 

Cloned directory from https://github.com/oulutan/Drone_FasterRCNN.

The author forked the maskrcnn repository, due to FAIR and trained a model on drone data.
Hence his work is an important baseline for us, since drone image segmentation is not straightforward
with convnets trained on 'normal' image segmentation tasks.
(We realised having a specific classifier head is essential - when we fed ground-truth boxes into
state-of the art classifiers, the accuracy was poor)

Consequently, we plain to do transfer learning on the above architecture (we do not have access to
huge GPUs, so this seems to be the only plausible option for us.)


Implemented substantial updates in the Dockefile.

## Usage

Usage: download the Dockefile and the weigths for the drone-trained newtwork from
 https://drive.google.com/file/d/1SCJf2JJmyCbxpDuy4njFaDw7xPqurpaQ/view?usp=sharing into your custom
directory

Build the docker image.
