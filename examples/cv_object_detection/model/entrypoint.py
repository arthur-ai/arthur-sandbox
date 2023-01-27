import os
import pathlib
import xml.etree.ElementTree as ET

from utils.utils import get_yolo_boxes
from utils.colors import get_color
import tensorflow as tf
import numpy as np
import cv2

# load saved model
path = pathlib.Path(__file__).parent.absolute()
model_path = "yolo_voc.h5"
model = tf.keras.models.load_model(f"{path}/{model_path}")

class_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def parse_annotations_voc(img_dir, ann_dir):
    '''
    Parses annotations given in XML format. Based on https://github.com/experiencor/keras-yolo3/blob/master/voc.py.
        
        Parameters:
            img_dir (str): Path to directory with raw images
            ann_dir (str): Path to directory with annotations in XML format

        Returns:
            all_insts (list): List wtih dictionary for each image
    '''

    all_insts = []

    for ann in sorted(os.listdir(ann_dir)):
        
        img = {'gt':[]}

        try:
            tree = ET.parse(ann_dir + ann)    
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = elem.text
            #     img['filepath'] = img_dir + elem.text
            # if 'width' in elem.tag:
            #     img['width'] = int(elem.text)
            # if 'height' in elem.tag:
            #     img['height'] = int(elem.text)
            
            if 'object' in elem.tag or 'part' in elem.tag:
                
                obj = {}
                img['gt'] += [obj]
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        
                        obj['label'] = attr.text
                        obj['box'] = {}

                    if 'bndbox' in attr.tag:
                        
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['box']['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['box']['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['box']['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['box']['ymax'] = int(round(float(dim.text)))

        all_insts += [img]

    return all_insts

def predict(image, net_h=416, net_w=416, obj_thresh=0.5, nms_thresh=0.45, labels=[], anchors=[]):
    '''
    Performs object detection using a given model. 
    Based on https://github.com/experiencor/keras-yolo3/blob/master/predict.py.

        Parameters:
            model (tf.keras.Model): Object detection model
            image (numpy.array): Raw image
            net_h (int): 
            net_w (int): 
            obj_thresh (float): 
            nms_thresh (float):
            labels (list):
            anchors (list):

        Returns:
            predictions (list[list]): List of bounding boxes for each image

    '''

    if labels == []:
        
        # default labels for VOC dataset
        labels = class_labels

    if anchors == []:

        # default anchors for VOC dataset
        anchors = [24,34, 46,84, 68,185, 116,286, 122,97, 171,180, 214,327, 326,193, 359,359]

    # instantiate an empty list
    predictions = []

    # get the bounding boxes
    boxes = get_yolo_boxes(model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

    # filter bounding boxes over a given score
    for box in boxes:
    
        label_idx = box.get_label()
        score = box.get_score()
            
        if score >= obj_thresh:

            pred = {}
            pred['box'] = {}

            pred['label'] = labels[label_idx]
            pred['score'] = score
            pred['box']['xmin'] = box.xmin
            pred['box']['ymin'] = box.ymin
            pred['box']['xmax'] = box.xmax
            pred['box']['ymax'] = box.ymax

            predictions += [pred]

    # convert box to form arthur expects
    new_boxes = []
    for box in predictions:
        class_id = class_labels.index(box['label'])
        conf = float(box['score'])
        x = box['box']['xmin']
        y = box['box']['ymin']
        width = box['box']['xmax'] - box['box']['xmin']
        height = box['box']['ymax'] - box['box']['ymin']
        new_boxes.append([class_id, conf, x, y, width, height])
    return new_boxes

def draw_boxes(raw_image, annotations):
    '''
    Adds given annotations to an image.

        Parameters:
            raw_image (numpy.array): Raw image
            annotations (list[list]): List of annotations for given image

        Returns:
            image (numpy.array): Annotated image

    '''

    image = raw_image.copy()

    for annot in annotations:
        
        label = annot[0]
        conf = annot[1]
        xmin = annot[2]
        ymin = annot[3]
        xmax = xmin + annot[4]
        ymax = ymin + annot[5]

        label_str = class_labels[label] + ' ' + str(round(conf, 4))

        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
        width, height = text_size[0][0], text_size[0][1]
        region = np.array([[xmin-3,        ymin], 
                            [xmin-3,        ymin-height-26], 
                            [xmin+width+13, ymin-height-26], 
                            [xmin+width+13, ymin]], dtype='int32')  

        cv2.rectangle(img=image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=get_color(label), thickness=5)
        cv2.fillPoly(img=image, pts=[region], color=get_color(label))
        cv2.putText(img=image, 
                    text=label_str, 
                    org=(xmin+2, ymin-13), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1.25e-3 * image.shape[0], 
                    color=(0,0,0), 
                    thickness=1)
                
    return image