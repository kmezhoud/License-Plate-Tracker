import cv2
import numpy as np
import pytesseract
import easyocr
import pyocr
import pyocr.builders
from PIL import Image  # Import the Image class
import re


def process_frame(frame, ocr_type):
  
    # Initialize variables
    tracking_started = False
    tracked_object = None

    # Load Tiny YOLO
    net = cv2.dnn.readNet("yolo/yolov3.cfg",
                          "yolo/yolov3.weights")
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getUnconnectedOutLayersNames()

    # Initialize OCR tool based on the selected type
    if ocr_type == 'tesseract':
        # Path to the Tesseract executable (update this path based on your installation)
        tesseract_path = '/usr/bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    elif ocr_type == 'easyocr':
        reader = easyocr.Reader(['en', 'ar'], gpu=True)  # Specify languages as needed
    elif ocr_type == 'pyocr':
        # Initialize PyOCR
        tools = pyocr.get_available_tools()
        tool = tools[0]  # Use the first available OCR tool
    else:
        raise ValueError("Invalid OCR type. Choose 'tesseract', 'easyocr', or 'pyocr'.")

    height, width, _ = frame.shape

    # Convert BGR to RGB
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Post-process the outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to "car" in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices: #.flatten()
        box = boxes[i]
        x, y, w, h = box
        
        if tracking_started:
              # Update the tracker with the new bounding box
              success, bbox = tracker.update(frame)
              if success:
                  x, y, w, h = [int(i) for i in bbox]
                  tracked_object = (x, y, w, h)
                  cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
  
                  # Crop the region of interest (ROI) for license plate recognition
                  roi = frame[y:y + h, x:x + w]
                  # Process the ROI using your OCR logic (omitted for brevity)
              else:
                  # Tracking failed, reset variables
                  tracking_started = False
                  tracked_object = None
        else:
            # Start tracking
            tracker.init(frame, (x, y, w, h))
            tracking_started = True
            tracked_object = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the region of interest (ROI) for license plate recognition
            roi = frame[y:y + h, x:x + w]

        # Use the selected OCR tool to extract text from the license plate
        if ocr_type == 'tesseract':
          # Convert the ROI to grayscale for better Tesseract OCR accuracy
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)
            custom_config = r'--oem 3 --psm 6 outputbase digits -l ara+eng --tessedit_char_whitelist=0123456789تونس'
             # Tesseract OCR
            txt = pytesseract.image_to_string(thresh, config=custom_config)
            confidence = confidences[i] * 100  # Confidence in percentage
        elif ocr_type == 'easyocr':
            # EasyOCR
            results = reader.readtext(roi)
            txt = results[0][1] if results else ""
            confidence = confidences[i] * 100  # Confidence in percentage
        elif ocr_type == 'pyocr':
            # Create an ImageTostringBuilder with custom configuration
           # custom_config = {
               # 'oem': 3,             # Set OCR Engine Mode to 3 (default)
                #'psm': 6,             # Set page segmentation mode to 6 (single block of text)
                #'outputbase': 'digits',  # Output recognition results in digits only
            #    lang='ara+eng',    # Specify languages for OCR (Arabic and English)
             #   'tessedit_char_whitelist': '0123456789تونس'  # Set whitelist of characters to recognize
           # }
           # builder = pyocr.builders.TextBuilder(**custom_config)
            # Use the custom configuration with the builder
            
            txt = tool.image_to_string(
                Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)),
                lang="ara+eng",
                builder= pyocr.builders.TextBuilder(tesseract_layout=6)
            )
            confidence = confidences[i] * 100  # Confidence in percentage
        else:
            raise ValueError("Invalid OCR type. Choose 'tesseract', 'easyocr', or 'pyocr'.")

            # Find and replace Arabic characters using a more flexible pattern
        #txt = re.sub(r'[^\w\s\d]', ' ', txt)  # Remove non-alphanumeric characters  تونس
        if 'تونس' in txt:
            # If 'تونس' is already present, leave the text unchanged
            #txt = re.sub(r'[^\w\s]', 'تونس', txt) 
            print("tunis is in text")
            #pass
        else:
            # Replace any sequence of characters with 'تونس'
            #txt+='تونس'
            print(f"{txt} (Confidence: {confidence:.2f}%)")
            #fps = float(cap.get(CAP_PROP_FPS))
            #print(fps)
            #pass
        # text to display    
        text_to_display = f"{txt} (Confidence: {confidence:.2f}%)"
       # Draw a filled black rectangle for the text background
        text_size = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y - 10 - text_size[1]), (x + text_size[0], y - 10), (0, 0, 0), -1)

        # Draw the license plate text on the frame
        cv2.putText(frame, text_to_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# Load video
cap = cv2.VideoCapture("../video/2180_16_M_20231117074029_motion.avi")

# Initialize object tracker (using MIL tracker)
tracker = cv2.TrackerMIL_create()

frame_count=0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % 5 == 0:
        continue
        
    # Resize the frame to reduce processing time
    #frame = cv2.resize(frame, (1200, 1000))
    #height, width, _ = frame.shape
    
        # Process the frame using PyOCR
    processed_frame = process_frame(frame, ocr_type='easyocr')  #'tesseract', 'easyocr' 'pyocr'
    
    cv2.imshow("Frame", processed_frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
