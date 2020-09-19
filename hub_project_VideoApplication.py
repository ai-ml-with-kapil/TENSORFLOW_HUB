import os
import pathlib
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

import cv2
cap = cv2.VideoCapture(0)


from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name+'.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar = True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print(model_dir)
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./models/research/object_detection/test_images')
#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./MyTestImages')
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print("*********** Model Part ************")
print("Inputs : ",detection_model.inputs)
print("Output Types : ",detection_model.output_dtypes)
print("Output Shapes : ",detection_model.output_shapes)
print("***********************************")

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis]

    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key, value in output_dict.items()}

    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.unit8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_np):
    #image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], category_index, instance_masks=output_dict.get('detection_masks_reframes', None), use_normalized_coordinates=True, line_thickness=8)
    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

while True:
  ret, image_np = cap.read()
  show_inference(detection_model, image_np)
  if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
