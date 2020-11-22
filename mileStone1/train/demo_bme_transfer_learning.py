# ! imports detection model
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo # ! helper class, which loads model from config file and
                               #   perform pre-processing, model prediction and post-processing
import cv2

config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_bme_transfer_learning.yaml"  # !get our network config

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "visdrone_model_0360000.pth"])  # ! get the pretrained weights

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('visdrone_test_img_0000001_02999_d_0000005.jpg')  # ! get an image
predictions = coco_demo.run_on_opencv_image(image) # ! get the prediction overlayed on the image
#cv2.imwrite('drone_res.jpg', predictions)
cv2.imshow('Predictions', predictions)
cv2.waitKey(0)
