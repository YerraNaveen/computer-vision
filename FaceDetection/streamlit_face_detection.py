import streamlit as st
import cv2
import numpy as np

@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

def processDetections(detections, image, selected_confidence):
    h, w  = image.shape[0:2]
    for face in range(detections.shape[2]):
        confidence = detections[0,0,face,2]
        if confidence > selected_confidence:
            x1, y1, x2, y2 = (detections[0,0,face,3:7] * np.array([w, h, w, h]))
            print("x1 :{}, y1: {}".format(x1, y1))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness = 2, lineType=cv2.LINE_AA)
    return image


st.title("Opencv ")
image_buf  = st.file_uploader("Choose an image with faces in it", type = ["jpg", "jpeg", "png"])
if image_buf is not None:
    raw_bytes = np.asarray(bytearray(image_buf.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    column = st.columns(2)
    column[0].image(image, channels="BGR")
    column[0].text("Input image")
    net = load_model()
    detections = detectFaceOpenCVDnn(net, image)
    confidence = st.slider("Set confidence", 0.1, 1.0, 0.5)

    processed_image = processDetections(detections, image, confidence)

    column[1].image(processed_image, channels="BGR")
    column[1].text("Output Image")