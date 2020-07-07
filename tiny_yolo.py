import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo_keras.utils import *
from yolo_keras.model import *

# Get the COCO classes on which the model was trained
classes_path = "yolo_keras/chagas_classes.txt"
with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names] 
num_classes = len(class_names)

# Get the anchor box coordinates for the model
anchors_path = "yolo_keras/chagas_anchors.txt"
with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

# Set the expected image size for the model
model_image_size = (608, 608)

# Create YOLO model
model_path ="yolo_keras/custom_tiny.h5"
yolo_model = load_model(model_path, compile=False)

# Generate output tensor targets for bounding box predictions
# Predictions for individual objects are based on a detection probability threshold of 0.3
# and an IoU threshold for non-max suppression of 0.45
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                    score_threshold=0.19, iou_threshold=0.10)

print("YOLO model ready!")
def detect_objects(image):
    
    # normalize and reshape image data
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # Predict classes and locations using Tensorflow session
    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    return out_boxes, out_scores, out_classes

def get_boxes(image, out_boxes,out_classes,out_scores):
    import random
    from PIL import Image
     
 
    # Plot the image
    img = np.array(image)

    # Set up padding for boxes
    img_size = model_image_size[0]
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    predicted_classes=[]
    boxes=[]
    scores=[]
    # process each instance of each class that was found
    for i, c in reversed(list(enumerate(out_classes))):

        # Get the class name
        predicted_class = class_names[c]
        predicted_classes.append(predicted_class)
        # Get the box coordinate and probability score for this instance
        box = out_boxes[i]
        score = out_scores[i]
        scores.append(score)
        # Format the label to be added to the image for this instance
        label = '{} {:.2f}'.format(predicted_class, score)

        # Get the box coordinates
        top, left, bottom, right = box
        y1 = max(0, np.floor(top + 0.5).astype('int32'))
        x1 = max(0, np.floor(left + 0.5).astype('int32'))
        y2 = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        x2 = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # Set the box dimensions
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        boxes.append([int(x1),int(y1),int(box_w),int(box_h)])

    return boxes, predicted_classes,scores
        


    
print("Functions ready")

import os
from PIL import Image

img_path = "348.jpg"

image = Image.open(img_path)

# Resize image for model input
image_data = letterbox_image(image, tuple(reversed(model_image_size)))

# Detect objects in the image
out_boxes, out_scores, out_classes = detect_objects(image_data)

# How many objects did we detect?
print('Found {} objects in {}'.format(len(out_boxes), img_path))

# Display the image with bounding boxes
boxes,predicted_classes,scores=get_boxes(image, out_boxes, out_classes,out_scores)
data={}
data["predictions"] = []
# for i in range(len(out_classes)):
#     box = out_boxes[i]
#     r={"label":predicted_classes[i],"probability":float(out_scores[i]),"x":box[0],"y":box[1],"w":box[2],"h":box[3]}
#     data["predictions"].append(r)
print(boxes,predicted_classes,scores)
