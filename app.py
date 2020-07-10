# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import tensorflow as tf
from keras import backend as K
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import render_template
import io
from keras.models import model_from_json
from numpy import expand_dims

import os
from keras.models import load_model
from keras.layers import Input
from yolo_keras.utils import *
from yolo_keras.model import *


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3 
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def decode_netout(netout, anchors, obj_thresh,  net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
app = flask.Flask(__name__)
# initialize our Flask application and the Keras model
# Get the COCO classes on which the model was trained
@app.route("/Yolo-tiny", methods=["POST"])
def tiny():
    
    json_file=open('yolo-tiny.json','r')     
    # carga el json y crea el modelo
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # se cargan los pesos (weights) en el nuevo modelo
    model.load_weights("yolo-tiny.h5")
    print("Modelo cargado desde el PC")
    # initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.2, 0.3
    anchors = [[10,14,  23,27,  37,58],  [81,82,  135,169,  344,319]]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
	# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
           
            image_h=image.size[0]
            image_w=image.size[1]
            print(image_h,image_w)
            image = image.resize((net_w , net_w ))
			# preprocess the image and prepare it for classification
            #image = prepare_image(image, target=(416, 416))
            image = img_to_array(image)
            # scale pixel values to [0, 1]
            image = image.astype('float32')
            image /= 255.0
            # add a dimension so that we have one sample
            image = expand_dims(image, 0)
			# classify the input image and then initialize the list
			# of predictions to return to the client
           
            yolos = model.predict(image)
           
            # summarize the shape of the list of arrays
            print([a.shape for a in yolos])

            # define the anchors
            anchors = [ [81,82,  135,169,  344,319],[10,14,  23,27,  37,58]]
            # define the probability threshold for detected objects
            class_threshold = 0.4
            boxes = list()

            for i in range(len(yolos)):
                    # decode the output of the network
                boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh,  net_h, net_w)
            
            # correct the sizes of the bounding boxes
            correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

            # suppress non-maximal boxes
            do_nms(boxes, nms_thresh)

            # get the details of the detected objects
            v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)


            data ["success"]=True
            data["predictions"] = []
            for i in range(len(v_labels)):
                box = v_boxes[i]
                r={"label":v_labels[i],"probability":float(v_scores[i]),"x_min":box.xmin,"y_min":box.ymin,"x_max":box.xmax,"y_max":box.ymax}
                data["predictions"].append(r)
                
    # return the data dictionary as a JSON response
    
    sess=K.get_session()
    K.clear_session()
    #return flask.jsonify(data)
    return render_template("nets.html",name="Yolo-tiny",predictions=data["predictions"])


@app.route("/Chagas", methods=["POST"])
def chagas():
	# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
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


            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # Resize image for model input
            image_data = letterbox_image(image, tuple(reversed(model_image_size)))

            # Detect objects in the image
                # normalize and reshape image data
            image_data = np.array(image_data, dtype='float32')
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

            # bounding boxes
             # Plot the image
            img = np.array(image_data)

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
            
            data={}
            data["predictions"] = []
            for i in range(len(out_classes)):
                box = boxes[i]
                r={"label":predicted_classes[i],"probability":float(scores[i]),"box":{"x":box[0],"y":box[1],"w":box[2],"h":box[3]}}
                data["predictions"].append(r)

    # return the data dictionary as a JSON response
    sess.clear_session()
    #return flask.jsonify(data)
    return render_template("nets.html",name="Chagas",predictions=data["predictions"])


@app.route("/ResNet50",methods=["POST"])
def routeResNet50():
   
    model = ResNet50(weights="imagenet")
   
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
           
            
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response

    sess=K.get_session()
    K.clear_session()
    #return flask.jsonify(data)
    return render_template("nets.html",name="ResNet50",predictions=data["predictions"])

@app.route("/nets/<string:id_net>",methods=["GET"])
def nets(id_net=None):
    return render_template("nets.html",name=id_net)
@app.route("/",methods=["GET"])
def main():
    return render_template("index.html")
@app.route("/index",methods=["GET"])
def index():
    return render_template("index.html")


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	#load_model()
	app.run(debug=True)