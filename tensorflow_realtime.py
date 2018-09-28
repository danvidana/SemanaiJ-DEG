import numpy as np
import os
import sys
import tensorflow as tf
import cv2

cap = cv2.VideoCapture(0)

sys.path.append("..")

os.chdir( 'D:/Tensorflow/models/research/object_detection/' )

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util



PATH_TO_CKPT ='C:/Users/juanc/Desktop/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
#PATH_TO_CKPT ='C:/Users/juanc/Desktop/inference_graph/frozen_inference_graph.pb'


PATH_TO_LABELS = "C:/Users/juanc/Desktop/ssd_mobilenet_v1_coco_2018_01_28/ssd.pbtxt"
#PATH_TO_LABELS = "C:/Users/juanc/Desktop/inference_graph/label_map.pbtxt"

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) == ord('q'):
                cv2.destroyAllWindows()
                break