# %%
# pip install torch torchvision torchaudio yolov5
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr  #create an interactive app need version 3.5.0 and httpx==0.24.1
import cv2
import requests
import torch 
from ultralytics import YOLO
import yolov5
import easyocr
from paddleocr import PaddleOCR
from paddleocr import draw_ocr
import keras_ocr
## needed by keras
import tensorflow as tf



# %%
# Loading Yolo V5 model
#model = yolov5.load('License_Plate_Model_Y5.pt')#, device="cpu")
model = yolov5.load('yolov5/best.pt')#, device="cpu")

# %%
def get_LicencePlate(frame, conf_threshold: gr.inputs.Slider = 0.1):
    
    lst_plate_xyxy = []
    # Setting model configuration 
    model.conf = conf_threshold

    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        class_name=(row['name'])
        if (class_name == 'license-plate'):
            lst_plate_xyxy.append((xmin,ymin,xmax,ymax, class_name))

    return lst_plate_xyxy

# %%
def extract_License_Number_PaddleOCR(image_path, conf_threshold):
    image = cv2.imread(image_path)
    # image = cv2.resize(image,(1020,800))
    results = get_LicencePlate(image, conf_threshold)
    print(f"results: {results}")
    PaddleOCR_output = []
    words = []
    boxes = []
    scores = []
    for index in results:
        x1 = index[0]
        y1 = index[1]
        x2 = index[2]
        y2 = index[3]
        class_name=(index[4])

        cropped_img = image[y1:y2, x1:x2]

        # processed_img = cropped_img
        processed_img =  cropped_img

        # Perform OCR using PaddleOCR
        paddle_ocr = PaddleOCR()
        paddle_ocr_result = paddle_ocr.ocr(processed_img)

        PaddleOCR_output.append(paddle_ocr_result)    

        print('\nPaddleOCR:')
        print(''.join([text[1][0] + ' ' for text in paddle_ocr_result[0]]))

        # Extract the words and bounding boxes from the OCR results
        #words = []
        #boxes = []
        #scores = []
        for line in paddle_ocr_result:
            for bbox in line:
                words.append(bbox[1][0])
                scores.append(bbox[1][1])
                boxes.append(bbox[0])
        
        output_image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(output_image,str(words),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

    return words, boxes, scores, output_image    

# %%
def extract_License_Number_EasyOCR(image_path,conf_threshold ):
    image = cv2.imread(image_path)
    # image = cv2.resize(image,(1020,800))
    results = get_LicencePlate(image ,conf_threshold=0.1  )
    easyOCR_output = []
    paddle_output = []

    # print("results************************", results)
    for index in results:
        x1 = index[0]
        y1 = index[1]
        x2 = index[2]
        y2 = index[3]
        class_name=(index[4])

        cropped_img = image[y1:y2, x1:x2]
        processed_img =  cropped_img

        easyocr_reader = easyocr.Reader(['en'], verbose=False)
        easyocr_result = easyocr_reader.readtext(processed_img)

        easyOCR_output.append(easyocr_result)     

        print('\nEasyOCR:')
        print(''.join([text[1] + ' ' for text in easyocr_result]))

        # Extract the words and bounding boxes from the OCR results
        words = []
        boxes = []
        scores = []
        for line in easyocr_result:
            words.append(line[1])
            scores.append(line[2])
            boxes.append(line[0])
        output_image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(output_image,str(words),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
  
    return words, boxes, scores, output_image

# %%
def createDataframe(words, boxes, scores):
    df = pd.DataFrame(list(zip(words, boxes, scores)), columns=['words', 'boxes', 'scores'])
    return df

# %%
def initiate_Extract(image_path, conf_threshold: gr.inputs.Slider = 0.10, ocr_type="PaddleOCR"):

    if ocr_type == "PaddleOCR":
        words, boxes, scores, output_img = extract_License_Number_PaddleOCR(image_path, conf_threshold)
    elif ocr_type == "EasyOCR":
        words, boxes, scores, output_img = extract_License_Number_EasyOCR(image_path, conf_threshold)
    else:
        words, boxes, scores, output_img = extract_License_Number_PaddleOCR(image_path ,conf_threshold )

    dataframe = createDataframe(words, boxes, scores)
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), dataframe


# %%
import numpy as np
title = "License Plate Number Recognition using YOLO & Paddle OCR - EasyOCR"
description = "Description of the App...."

css = """.output_image, .input_image {height: 600px !important}"""
examples = [['../video/truck1.JPG'],['../video/c1.jpg'], ['../video/195724.png']]
iface = gr.Interface(fn=initiate_Extract,
                     inputs=[
                        gr.inputs.Image(type="filepath", label="Input Image"),
                        gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.10, step=0.05, label="Confidence Threshold"),
                        gr.inputs.Dropdown(label="Select the OCR",default="PaddleOCR", choices=["PaddleOCR", "EasyOCR"]),
                     ],
                     outputs=[gr.outputs.Image(type="pil", label="annotated image"),"dataframe"] ,
                     title=title,
                     description=description,
                     examples=examples,
                     css=css,
                     analytics_enabled = True, enable_queue=True)

iface.launch(inline=False , debug=True, share=True)

# %%


# %%



