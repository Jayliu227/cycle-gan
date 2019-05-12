import numpy as np
import os
import six.moves.urllib as urllib
import sys

import tensorflow as tf
import torch

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# disable CPU AVX support warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'object_detection/mask_rcnn_inception_v2_coco_2018_01_28'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def shift_img_to_upper_left(a):
    while np.count_nonzero(a[0,:]) == 0:
        a = np.concatenate((a[1:,],a[:1,]))
    while np.count_nonzero(a[:,0]) == 0:
        a = np.concatenate((a[:,1:],a[:,:1]), axis=1)
    return a 

def get_img_classification(img):
    '''    
    input:
        4d np array of size (1, H, W, 3)
    output:
        a dictionary including: 
        ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']
    '''
    output_dict = run_inference_for_single_image(img, detection_graph)
    return output_dict

def get_mask(img):
    '''
    input:
        4d pytorch tensor image (1, 3, H, W)
    output:
        2d np array representing mask where 0 is background and 1 is foreground
    '''
    # convert pytorch tensor to np array
    img = img.numpy() * 255
    
    # change dimension from (1, 3, H, W) to (1, H, W, 3)
    img = np.transpose(img, (0, 2, 3, 1))
    
    # get segmentation
    output_dict = get_img_classification(img)
    
    # return the mask that has largest area, as there are several masks 
    if len(output_dict['detection_masks']) == 0:
        # print('Error: Did not detect any shape')
        return np.zeros((img.shape[1], img.shape[2]))        
    return output_dict['detection_masks'][np.argmax([np.count_nonzero(mask) for mask in output_dict['detection_masks']])]


def shape_sim(mask_a, mask_b):
    '''
    input:
        two 2d np array, image masks
    output:
        a scored determined by their area overlapping
    '''
    a = np.count_nonzero(np.minimum(mask_a,mask_b))
    b = max(np.count_nonzero(mask_a), np.count_nonzero(mask_b))
    return a / max(1e-5, b)
