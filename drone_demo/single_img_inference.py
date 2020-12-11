# in this module, inference on a single picture is performed

# this contains predictions with the model, and drawing the predicted vs. ground-truth boxes 

# this code is literally a patchwork in the sense that it contains code copied from many locations
# (eg. from zold137/drone_demo/predictor

import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
#from maskrcnn_benchmark.data.transforms import build_transformsioi


def evalpipe(image,predictions,threshold):
    '''
    input:
    image: np.ndarray - assuming that the size to what BoxList corresponds is the same as the image size!
    predictions: BoxList object, containing 'labels' and 'scores' fields
    threshold: boxes of higher confidency are drawn
    '''
    
    to_draw=select_top_predictions(threshold,predictions)
    image=overlay_boxes(image,to_draw)
    image=overlay_class_names(image,to_draw)
    
    return image
    

def select_top_predictions(threshold, predictions):
    """
    Select only predictions which have a `score` > threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    
    colors = labels[:, None] *palette # self.palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions,gt=False):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        
        if not gt:
        
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )
        else:
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255,0,0), 2
            )

    return image

def overlay_class_names(image, predictions,gt=False):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        
        
        id_to_object={ 8:'1F',
                             1:'1B',
                            2:'1L',
                             3:'1R',
                            4:'2' ,
                            5:'5H',
                             6:'5L',
                            7:'0',
                             0:'00' }
        object_to_cat={
                    '1F': 'Front View',
                    '1B': 'Back View',
                    '1L': 'Left View',
                    '1R': 'Right View',
                    '2': ' Bicycle Crowd',
                    '5H': 'High-Density Human Crowd',
                    '5L': 'Low-Density Human Crowd',
                    '0': 'irrelevant TV graphics',
                   '00':'__background'}
        
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [object_to_cat[id_to_object[i]] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            
            if not gt:
            
                cv2.putText(
                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                )
            else:
                cv2.putText(
                    image, 'gt: '+s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                )

        return image
