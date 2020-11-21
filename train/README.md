# Train

### 1
e2e_faster_rcnn_X_101_32x8d_FPN_1x_bme_transfer_learning.yaml
  - can be defined the parameters of the neural network
  - This file exaclty describe a CN() object, of which parameters located in ./maskrcnn-benchmark/maskrcnn_benchmark/config/defaults.py
  - WEIGHT: "" will find the detecron model in ./maskrcnn-benchmark/maskrcnn_benchmark/config/paths_catalog.py
  - FREEZE_CONV_BODY_AT: 5 start train from stage 5
  - Datasets: put your own dataset
    - Train:
    - Test: 
  
### 2
demo_bme_transfer_learning.py
  - import the above refert CN() object as cfg, and give to the COCODemo helper class
  
  
  
### Extras

  ##### First
  Transfer learning freeze layers issue
  https://github.com/facebookresearch/maskrcnn-benchmark/issues/201
  
  Q"If i want to train Res101-FPN last four stage, what should I do? Set FREEZE_CONV_BODY_AT = 5?
  
  A"As you can see from here(https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/resnet.py#L111),
    if you want to only train starting from stage 4, you should set FREEZE_CONV_BODY_AT = 4 "
    
  ##### Second
  Freeze the RPN (sadly just in detectron)
  https://github.com/facebookresearch/Detectron/issues/143
  
  "Just insert a StopGradient op in the correct place(s). For example usage, see https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/ResNet.py#L105"
  
  
  ##### Third
  https://github.com/facebookresearch/maskrcnn-benchmark/issues/521
  Step-by-step tutorial train your own dataset
